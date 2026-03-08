"""
nodes/vision.py — VLM frame analysis node.

Sends extracted frames to Claude Vision and scores each one across:
  - Text accuracy     (on-screen text readable and correct?)
  - Visual quality    (sharp, well-composed, well-lit?)
  - Content safety    (anything inappropriate or brand-unsafe?)
  - Consistency       (does it match prior frames?)
  - Authenticity      (does this feel human and real, or generic AI slop?)

Samples frames intelligently — keyframes + scene boundaries.
"""

import base64
import json
import os
import re
from typing import Any
import anthropic

MAX_FRAMES = int(os.environ.get("MAX_VISION_FRAMES", 8))


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def sample_frames(frames: list[str], scene_cuts: list[float], max_frames: int) -> list[tuple[int, str]]:
    if not frames:
        return []
    total = len(frames)
    selected = set([0, total - 1])
    for cut in scene_cuts:
        selected.add(min(int(cut), total - 1))
        if len(selected) >= max_frames:
            break
    if len(selected) < max_frames:
        step = total // (max_frames - len(selected) + 1)
        for i in range(0, total, max(step, 1)):
            selected.add(i)
            if len(selected) >= max_frames:
                break
    return [(idx, frames[idx]) for idx in sorted(selected)[:max_frames]]


def _first_balanced_json_snippet(text: str):
    """Return the first balanced JSON object/array substring, if present."""
    if not text:
        return None

    for start_idx, start_ch in enumerate(text):
        if start_ch not in "[{":
            continue
        stack = []
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                open_ch = stack.pop()
                if (open_ch == "[" and ch != "]") or (open_ch == "{" and ch != "}"):
                    break
                if not stack:
                    return text[start_idx:i + 1]
    return None


def _parse_json_response(raw: str):
    """Best-effort JSON parser for model output that may include wrappers."""
    if raw is None:
        return None

    text = raw.strip()
    candidates = []

    if text:
        candidates.append(text)

    # Extract fenced code blocks (```json ... ``` or ``` ... ```)
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(fenced)

    snippet = _first_balanced_json_snippet(text)
    if snippet:
        candidates.append(snippet)

    for candidate in candidates:
        payload = candidate.strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    return None


def _extract_text_blocks(response: Any) -> str:
    """Join all Anthropic text blocks into one parseable string."""
    blocks = getattr(response, "content", None) or []
    texts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return "\n".join(texts).strip()


def _coerce_frame_scores(parsed: Any) -> list[dict] | None:
    """Normalize model payload into a frame-score list."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("frames", "frame_scores", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    return None


def analyze_frames(frames: list[str], scene_cuts: list[float], client: anthropic.Anthropic) -> list[dict]:
    sampled = sample_frames(frames, scene_cuts, MAX_FRAMES)
    if not sampled:
        return []

    def _build_content(samples: list[tuple[int, str]], compact: bool) -> list[dict]:
        issue_cap = "Keep each 'issues' array to max 2 short strings." if compact else ""
        content = [
            {
                "type": "text",
                "text": (
                    "You are a professional video quality critic for an AI video generation platform. "
                    "Your job is to catch everything that makes a video feel like 'AI slop' — generic, "
                    "inauthentic, template-like, or technically broken.\n\n"
                    "Analyze each frame below across these 5 dimensions:\n\n"
                    "1. **text_accuracy** — Is any on-screen text readable? Correct language? No gibberish, typos, or wrong words?\n"
                    "2. **visual_quality** — Is the frame sharp, well-composed, properly lit? Any glitches, artifacts, warping?\n"
                    "3. **content_safety** — Is there anything inappropriate, offensive, or brand-unsafe?\n"
                    "4. **consistency** — Does the visual style, color palette, characters or products match what you'd "
                    "expect in a coherent, continuous video?\n"
                    "5. **authenticity** — This is critical. Does this frame feel like it was made by a real creator "
                    "with intent, or does it feel generic and AI-generated? Look for: stock-photo energy, "
                    "unnatural poses, plastic-looking faces, overly perfect lighting, template layouts, "
                    "soulless composition. A score of 100 means it could fool anyone into thinking a real "
                    "human creator made it with care. A score below 50 means it screams 'AI slop'.\n\n"
                    "For each frame respond with a JSON array where each element has:\n"
                    "- frame_index (int)\n"
                    "- timestamp_seconds (int)\n"
                    "- text_accuracy: { score: 0-100, issues: [string] }\n"
                    "- visual_quality: { score: 0-100, issues: [string] }\n"
                    "- content_safety: { score: 0-100, issues: [string] }\n"
                    "- consistency: { score: 0-100, issues: [string] }\n"
                    "- authenticity: { score: 0-100, issues: [string], feels_like: 'one phrase describing the vibe' }\n"
                    "- overall_frame_score: 0-100\n\n"
                    f"{issue_cap}\n"
                    "Return ONLY valid JSON. No markdown, no preamble."
                )
            }
        ]
        for idx, frame_path in samples:
            content.append({"type": "text", "text": f"Frame at {idx}s:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image(frame_path),
                }
            })
        return content

    attempts = [
        {"samples": sampled, "max_tokens": 4000, "compact": False},
        {"samples": sampled[: max(1, len(sampled) // 2)], "max_tokens": 3000, "compact": True},
    ]

    for i, attempt in enumerate(attempts, start=1):
        samples = attempt["samples"]
        print(f"[vision] Sending {len(samples)} frames to Claude Vision...")
        response = client.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=attempt["max_tokens"],
            temperature=0,
            messages=[{"role": "user", "content": _build_content(samples, attempt["compact"])}]
        )

        raw = _extract_text_blocks(response)
        parsed = _parse_json_response(raw)
        frame_scores = _coerce_frame_scores(parsed)
        if frame_scores is not None:
            return frame_scores

        stop_reason = getattr(response, "stop_reason", None)
        preview = (raw[:600] + "...") if raw and len(raw) > 600 else raw
        print(f"[vision] Warning: could not parse VLM response as JSON (attempt {i}/{len(attempts)})")
        if stop_reason:
            print(f"[vision] stop_reason={stop_reason}")
        if preview:
            print(f"[vision] Raw preview: {preview!r}")

    return []


# ─── LangGraph Node ───────────────────────────────────────────────────────────

def vision_node(state: dict) -> dict:
    client = anthropic.Anthropic()
    frame_scores = analyze_frames(
        state.get("frames", []),
        state.get("scene_cuts", []),
        client
    )
    if frame_scores:
        avg = sum(f.get("overall_frame_score", 0) for f in frame_scores) / len(frame_scores)
        auth_avg = sum(f.get("authenticity", {}).get("score", 0) for f in frame_scores) / len(frame_scores)
        print(f"[vision] {len(frame_scores)} frames | Avg score: {avg:.0f}/100 | Authenticity: {auth_avg:.0f}/100")
    else:
        print(f"[vision] No frame scores returned")
    return {**state, "frame_scores": frame_scores}
