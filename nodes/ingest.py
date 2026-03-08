"""
nodes/ingest.py — FFmpeg extraction node.

Extracts:
  - 1 frame per second (JPG) for VLM analysis
  - Keyframes for scene cut detection
  - Audio as 16kHz mono WAV for Whisper
  - Full metadata via ffprobe (duration, fps, resolution, codec)
  - Scene cut timestamps
"""

import os
import json
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state_types import CriticState


FRAMES_DIR = "output/frames"
SCENES_DIR = "output/scenes"


def run(cmd: list[str]) -> str:
    """Run a subprocess command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def extract_metadata(video_path: str) -> dict:
    """Use ffprobe to get video metadata as JSON."""
    out = run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path
    ])
    try:
        data = json.loads(out)
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {}
        )
        fmt = data.get("format", {})
        return {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")),  # e.g. "30/1" → 30.0
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "codec": video_stream.get("codec_name"),
            "format": fmt.get("format_name"),
        }
    except Exception:
        return {}


def extract_frames(video_path: str, output_dir: str) -> list[str]:
    """Extract 1 frame per second as JPG. Returns sorted list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "quiet",
        "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "2",
        f"{output_dir}/%04d.jpg"
    ])
    frames = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".jpg")
    ])
    return frames


def extract_audio(video_path: str, output_path: str) -> str | None:
    """Extract audio as 16kHz mono WAV for Whisper. Returns path or None."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = subprocess.run([
        "ffmpeg", "-y", "-v", "quiet",
        "-i", video_path,
        "-ar", "16000", "-ac", "1",
        output_path
    ])
    return output_path if result.returncode == 0 else None


def detect_scene_cuts(video_path: str) -> list[float]:
    """Detect scene cut timestamps using ffmpeg scene filter."""
    result = subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", "select='gt(scene,0.35)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ], capture_output=True, text=True)

    timestamps = []
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            try:
                part = [p for p in line.split() if p.startswith("pts_time:")][0]
                timestamps.append(float(part.replace("pts_time:", "")))
            except (IndexError, ValueError):
                continue
    return sorted(timestamps)


# ─── LangGraph Node ───────────────────────────────────────────────────────────

def ingest_node(state: dict) -> dict:
    """
    LangGraph node: extract all raw data from the video using FFmpeg.
    """
    video_path = state["video_path"]
    base = os.path.splitext(os.path.basename(video_path))[0]

    frames_dir = f"output/frames/{base}"
    audio_path = f"output/audio/{base}.wav"

    print(f"[ingest] Extracting metadata...")
    metadata = extract_metadata(video_path)
    print(f"[ingest] Duration: {metadata.get('duration', '?')}s | "
          f"Resolution: {metadata.get('width')}x{metadata.get('height')} | "
          f"FPS: {metadata.get('fps', '?')}")

    print(f"[ingest] Extracting frames (1fps)...")
    frames = extract_frames(video_path, frames_dir)
    print(f"[ingest] Extracted {len(frames)} frames")

    print(f"[ingest] Extracting audio...")
    audio = extract_audio(video_path, audio_path)
    print(f"[ingest] Audio: {'OK' if audio else 'X (no audio track)'}")

    print(f"[ingest] Detecting scene cuts...")
    cuts = detect_scene_cuts(video_path)
    print(f"[ingest] Found {len(cuts)} scene cuts at: {[round(t, 1) for t in cuts]}")

    return {
        **state,
        "frames": frames,
        "audio_path": audio,
        "metadata": metadata,
        "scene_cuts": cuts,
    }
