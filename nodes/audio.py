"""
nodes/audio.py — Audio transcription node.

Uses OpenAI Whisper (local, no API cost) to transcribe the extracted WAV.
The transcript feeds into the critic node for:
  - Audio/caption sync checking
  - Language consistency
  - Dead air / silence detection
"""

import os


def _stabilize_print_for_numba() -> None:
    """
    Numba expects callable globals (including print) to be reachable as
    module attributes. Some runners wrap print with a local function like
    __main__.captured_print, which breaks that assumption during import.
    """
    import builtins
    import sys

    wrapped_print = getattr(builtins, "print", None)
    if not callable(wrapped_print):
        return

    mod_name = getattr(wrapped_print, "__module__", None)
    attr_name = getattr(wrapped_print, "__name__", None)
    if not mod_name or not attr_name:
        return

    mod = sys.modules.get(mod_name)
    if mod is None or hasattr(mod, attr_name):
        return

    try:
        setattr(mod, attr_name, wrapped_print)
    except Exception:
        # Best-effort compatibility shim; safe to continue if it fails.
        pass


def transcribe_audio(audio_path: str) -> str | None:
    """
    Transcribe audio using Whisper (local model).
    Falls back gracefully if Whisper is not installed.
    """
    try:
        _stabilize_print_for_numba()
        import whisper
        print(f"[audio] Loading Whisper model (base)...")
        model = whisper.load_model("base")
        print(f"[audio] Transcribing {os.path.basename(audio_path)}...")
        # fp16=False for CPU compatibility; avoid multiprocessing issues on Windows
        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=False,
        )
        return result.get("text", "").strip()
    except ImportError:
        print(f"[audio] Whisper not installed — skipping transcription")
        print(f"[audio] Install with: pip install openai-whisper")
        return None
    except Exception as e:
        import traceback
        print(f"[audio] Transcription failed: {e}")
        traceback.print_exc()
        return None


def detect_silence(audio_path: str, silence_threshold_db: int = -35, min_duration: float = 2.0) -> list[dict]:
    """
    Use ffmpeg silencedetect to find stretches of silence/dead air.
    Returns list of {start, end, duration} dicts.
    """
    import subprocess

    result = subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_duration}",
        "-f", "null", "-"
    ], capture_output=True, text=True)

    silences = []
    current_start = None

    for line in result.stderr.split("\n"):
        if "silence_start" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip())
            except (IndexError, ValueError):
                pass
        elif "silence_end" in line and current_start is not None:
            try:
                parts = line.split("silence_end:")[1].split("|")
                end = float(parts[0].strip())
                duration = float(parts[1].replace("silence_duration:", "").strip())
                silences.append({
                    "start": round(current_start, 2),
                    "end": round(end, 2),
                    "duration": round(duration, 2)
                })
                current_start = None
            except (IndexError, ValueError):
                pass

    return silences


# ─── LangGraph Node ───────────────────────────────────────────────────────────

def audio_node(state: dict) -> dict:
    """
    LangGraph node: transcribe audio and detect silence.
    """
    audio_path = state.get("audio_path")

    if not audio_path or not os.path.exists(audio_path):
        print("[audio] No audio track found — skipping")
        return {**state, "transcript": None}

    transcript = transcribe_audio(audio_path)

    silences = detect_silence(audio_path)
    if silences:
        segments = [f"{s['start']}s-{s['end']}s" for s in silences]
        print(f"[audio] Found {len(silences)} silence segment(s): {segments}")

    if transcript:
        preview = transcript[:100] + "..." if len(transcript) > 100 else transcript
        print(f"[audio] Transcript: \"{preview}\"")

    # Attach silence data to metadata for the critic node
    metadata = state.get("metadata", {})
    metadata["silence_segments"] = silences

    return {
        **state,
        "transcript": transcript,
        "metadata": metadata,
    }
