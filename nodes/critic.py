"""
nodes/critic.py — LLM critic aggregation node.

Aggregates all pipeline signals into a final structured report with:
  - Overall quality score
  - Authenticity score  (does this feel human or AI slop?)
  - Audience fit score  (does it work for the target audience?)
  - Verdict: pass | refine | reject
  - Actionable issues with recommendations
"""

import json
import os
import anthropic

from state_types import AUDIENCE_OPTIONS


CRITIC_SYSTEM_PROMPT = """
You are a senior video quality critic for an AI video generation platform.
Your role is to catch everything that makes a video fail — technically, creatively, and for its audience.

You evaluate three things:
1. TECHNICAL QUALITY — text accuracy, visual quality, pacing, consistency, safety
2. AUTHENTICITY — does this feel like a real creator made it with intent?
   The enemy is "AI slop": generic stock-photo energy, unnatural poses, plastic faces,
   template layouts, soulless composition, robotic narration, content that could apply
   to any video about anything. A great video feels specific, intentional, human.
3. AUDIENCE FIT — does this actually work for the intended audience?
   Wrong tone, wrong length, wrong hooks, wrong vocabulary all count as failures.

Verdict guidelines:
- "pass"   → overall_score >= 80 AND authenticity_score >= 70, no high-severity issues
- "refine" → overall_score 50-79, OR authenticity_score 50-69, OR medium issues only
- "reject" → overall_score < 50, OR authenticity_score < 50, OR any high-severity issue

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.

JSON schema:
{
  "overall": "pass" | "fail",
  "overall_score": 0-100,
  "authenticity_score": 0-100,
  "audience_fit_score": 0-100,
  "verdict": "pass" | "refine" | "reject",
  "retry": true | false,
  "summary": "one punchy sentence — what is the single biggest problem or strength?",
  "authenticity_notes": "one sentence — what makes this feel AI-generated or human?",
  "audience_notes": "one sentence — how well does this serve the target audience?",
  "issues": [
    {
      "type": "text_accuracy" | "visual_quality" | "pacing" | "consistency" | "content_safety" | "authenticity" | "audience_fit",
      "severity": "high" | "medium" | "low",
      "timestamp": "HH:MM:SS",
      "description": "specific, concrete description of the problem",
      "recommendation": "specific, actionable fix — what exactly to change"
    }
  ]
}
"""


def build_critic_prompt(state: dict) -> str:
    metadata    = state.get("metadata", {})
    frame_scores = state.get("frame_scores", [])
    transcript  = state.get("transcript")
    scene_cuts  = state.get("scene_cuts", [])
    silences    = metadata.get("silence_segments", [])
    audience    = state.get("audience", "general")
    audience_desc = AUDIENCE_OPTIONS.get(audience, audience)

    lines = ["Here is the video analysis data:\n"]

    # Audience context — this shapes the entire evaluation
    lines.append("## Target Audience")
    lines.append(f"- Audience type: {audience}")
    lines.append(f"- Description: {audience_desc}")
    lines.append("- Evaluate tone, length, hooks, vocabulary, and storytelling against this audience.")
    lines.append("")

    # Metadata
    lines.append("## Video Metadata")
    lines.append(f"- Duration: {metadata.get('duration', 'unknown')}s")
    lines.append(f"- Resolution: {metadata.get('width')}x{metadata.get('height')}")
    lines.append(f"- FPS: {metadata.get('fps')}")
    lines.append("")

    # Pacing signals
    lines.append("## Pacing Signals")
    lines.append(f"- Scene cuts at: {[round(t, 1) for t in scene_cuts]}s")
    if silences:
        lines.append(f"- Dead air / silence: {silences}")
    else:
        lines.append("- No significant silence detected")
    lines.append("")

    # Frame scores including authenticity
    if frame_scores:
        lines.append("## Frame Analysis (VLM scores)")
        for f in frame_scores:
            ts = f.get("timestamp_seconds", "?")
            overall = f.get("overall_frame_score", "?")
            lines.append(f"\nFrame at {ts}s (overall: {overall}/100):")
            for dim in ["text_accuracy", "visual_quality", "content_safety", "consistency"]:
                d = f.get(dim, {})
                lines.append(f"  - {dim}: {d.get('score','?')}/100" +
                              (f" — {d.get('issues')}" if d.get("issues") else ""))
            auth = f.get("authenticity", {})
            lines.append(f"  - authenticity: {auth.get('score','?')}/100"
                         + (f" — feels like: '{auth.get('feels_like','')}'" if auth.get("feels_like") else "")
                         + (f" — {auth.get('issues')}" if auth.get("issues") else ""))
    else:
        lines.append("## Frame Analysis\nNo frame data available.")
    lines.append("")

    # Transcript
    lines.append("## Audio Transcript")
    if transcript:
        lines.append(f'"{transcript}"')
        lines.append("- Evaluate: does this sound like a real human or a robotic AI narrator?")
        lines.append("- Does the language fit the target audience?")
        lines.append("- Is there a clear hook, narrative arc, and payoff?")
    else:
        lines.append("No audio transcript available.")
    lines.append("")

    lines.append("Now produce the final critic report as JSON.")
    return "\n".join(lines)


# ─── LangGraph Node ───────────────────────────────────────────────────────────

def critic_node(state: dict) -> dict:
    client = anthropic.Anthropic()
    prompt = build_critic_prompt(state)

    audience = state.get("audience", "general")
    print(f"[critic] Evaluating for audience: '{audience}'...")
    print(f"[critic] Aggregating signals and generating report...")

    response = client.messages.create(
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=2000,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        report = json.loads(raw.strip())
    except json.JSONDecodeError:
        print("[critic] Warning: could not parse report as JSON")
        report = {"raw": raw, "verdict": "unknown", "overall_score": 0}

    # Normalise score key (support both "score" and "overall_score")
    if "score" not in report and "overall_score" in report:
        report["score"] = report["overall_score"]

    verdict        = report.get("verdict", "unknown")
    score          = report.get("score", 0)
    auth_score     = report.get("authenticity_score", 0)
    audience_score = report.get("audience_fit_score", 0)
    issues         = report.get("issues", [])

    print(f"\n{'='*55}")
    print(f"  VERDICT:      {verdict.upper()}")
    print(f"  SCORE:        {score}/100")
    print(f"  AUTHENTICITY: {auth_score}/100")
    print(f"  AUDIENCE FIT: {audience_score}/100  (target: {audience})")
    print(f"  ISSUES:       {len(issues)} found")
    print(f"  SUMMARY:      {report.get('summary', '')}")
    print(f"{'='*55}\n")

    return {**state, "report": report}