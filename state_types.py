"""
types.py — Shared state and data models for the critic pipeline.
"""

from typing import TypedDict, Optional
from dataclasses import dataclass, field


# ─── Audience Options ─────────────────────────────────────────────────────────

AUDIENCE_OPTIONS = {
    "general":     "General audience — casual viewers, social media",
    "technical":   "Technical audience — developers, engineers, product teams",
    "buyer":       "Buyer / decision-maker — wants ROI, outcomes, credibility",
    "casual":      "Casual / entertainment — TikTok, Reels, short-form",
    "educational": "Educational — learners, students, how-to seekers",
}


# ─── LangGraph State ──────────────────────────────────────────────────────────

class CriticState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Input
    video_path: str
    audience: str                # One of AUDIENCE_OPTIONS keys (default: "general")

    # Ingest node outputs
    frames: list[str]            # Paths to extracted JPG frames
    audio_path: Optional[str]    # Path to extracted WAV
    metadata: dict               # ffprobe JSON (duration, fps, resolution, etc.)
    scene_cuts: list[float]      # Timestamps (seconds) of detected scene cuts

    # Vision node outputs
    frame_scores: list[dict]     # Per-frame VLM analysis

    # Audio node outputs
    transcript: Optional[str]    # Whisper transcription

    # Critic node outputs
    report: Optional[dict]       # Final structured report


# ─── Report Models ────────────────────────────────────────────────────────────

@dataclass
class Issue:
    type: str           # text_accuracy | pacing | consistency | safety | quality | authenticity | audience_fit
    severity: str       # high | medium | low
    timestamp: str      # HH:MM:SS
    description: str    # What's wrong
    recommendation: str # What to do about it


@dataclass
class CriticReport:
    overall: str                        # pass | fail
    score: int                          # 0-100
    verdict: str                        # reject | refine | pass
    authenticity_score: int = 0         # 0-100 - how human/real does this feel?
    audience_fit_score: int = 0         # 0-100 - does it work for the target audience?
    issues: list[Issue] = field(default_factory=list)
    retry: bool = False
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "score": self.score,
            "verdict": self.verdict,
            "authenticity_score": self.authenticity_score,
            "audience_fit_score": self.audience_fit_score,
            "retry": self.retry,
            "summary": self.summary,
            "issues": [
                {
                    "type": i.type,
                    "severity": i.severity,
                    "timestamp": i.timestamp,
                    "description": i.description,
                    "recommendation": i.recommendation,
                }
                for i in self.issues
            ],
        }