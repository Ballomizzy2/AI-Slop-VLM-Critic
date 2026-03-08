# Critic Pipeline

> An AI critic that reviews generated videos and outputs structured quality reports.
> Evaluates technical quality, **authenticity** (human vs AI slop), and **audience fit**.
> Built for the OpusClip Hackathon — Problem 5: Critic in the Loop.

---

## What it does

```
video.mp4 → [ffmpeg] → frames + audio + metadata
          → [Claude Vision] → frame scores (incl. authenticity)
          → [Whisper] → transcript
          → [Claude LLM] → structured report
```

**Output:**
```json
{
  "overall": "fail",
  "overall_score": 61,
  "authenticity_score": 45,
  "audience_fit_score": 72,
  "verdict": "refine",
  "retry": false,
  "summary": "On-screen text is garbled at 4s; dead air at 12s.",
  "authenticity_notes": "Feels generic AI slop — plastic faces, template layout.",
  "audience_notes": "Tone too casual for technical audience.",
  "issues": [
    {
      "type": "text_accuracy",
      "severity": "high",
      "timestamp": "00:00:04",
      "description": "On-screen text appears garbled",
      "recommendation": "Fix caption text at 4s mark before render"
    },
    {
      "type": "authenticity",
      "severity": "medium",
      "timestamp": "00:00:12",
      "description": "Frame feels like stock-photo — unnatural poses",
      "recommendation": "Regenerate with more human, intentional composition"
    }
  ]
}
```

---

## Setup

```bash
# 1. Install ffmpeg (system-level)
# Mac:
brew install ffmpeg
# Windows:
winget install ffmpeg

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## UI (Recommended)

```bash
# Start the server + UI
python server.py
# Open http://localhost:7474
```

Drop a video, watch the pipeline run in the terminal, see the report.  
**Report tab:** Video plays on the left; click any issue to seek the video to that timestamp.

---

## CLI Usage

```bash
# Basic usage
python main.py video.mp4

# Custom output directory
python main.py video.mp4 --output ./results

# Target audience (shapes authenticity + audience fit evaluation)
python main.py video.mp4 --audience casual

# Raw JSON output
python main.py video.mp4 --json
```

---

## Agent Endpoint

For machine-to-machine integration (e.g. from a video generation pipeline):

```bash
POST http://localhost:7474/api/evaluate
Content-Type: application/json

{
  "video_path": "/path/to/video.mp4",
  "audience": "casual"
}
```

**Response:** The full critic report JSON (synchronous — blocks until pipeline completes).

No upload needed — pass a path the server can access. Use for agents, CI, or other workflows.

---

## What it evaluates

| Dimension | Signal Source | What it checks |
|---|---|---|
| **Text accuracy** | Claude Vision | Readable? Correct language? No gibberish? |
| **Visual quality** | Claude Vision | Sharp, composed, well-lit? |
| **Pacing** | FFmpeg scene cuts + silence | Too fast? Dead air? |
| **Consistency** | Claude Vision | Same style/character across shots? |
| **Content safety** | Claude Vision | Anything inappropriate or brand-unsafe? |
| **Authenticity** | Claude Vision + Critic | Does it feel human vs AI slop? |
| **Audience fit** | Critic | Does it work for the target audience? |
| **Audio/transcript** | Whisper | Speech present? Matches captions? |

---

## Audience options

| Key | Description |
|---|---|
| `general` | General audience — casual viewers, social media |
| `technical` | Technical audience — developers, engineers, product teams |
| `buyer` | Buyer / decision-maker — wants ROI, outcomes, credibility |
| `casual` | Casual / entertainment — TikTok, Reels, short-form |
| `educational` | Educational — learners, students, how-to seekers |

The critic evaluates tone, length, hooks, vocabulary, and storytelling against the chosen audience.

---

## Verdicts

| Verdict | Meaning | When |
|---|---|---|
| `pass` | Ship it | overall_score ≥ 80 AND authenticity_score ≥ 70, no high-severity issues |
| `refine` | Fix specific issues | overall_score 50–79, OR authenticity_score 50–69, OR medium issues only |
| `reject` | Regenerate | overall_score < 50, OR authenticity_score < 50, OR any high-severity issue |

---

## Architecture

```
LangGraph Pipeline:

ingest_node  →  vision_node  →  audio_node  →  critic_node
   │               │               │               │
ffmpeg          Claude Vision    Whisper         Claude LLM
frames/audio    frame scores     transcript      final report
                + authenticity
```

Each node is independently replaceable — swap Whisper for another STT,  
swap Claude for another VLM, or add new nodes without touching existing ones.



<img width="1245" height="521" alt="image" src="https://github.com/user-attachments/assets/b0a668ed-433b-4ee8-8841-bb9f6a8500c6" />

---

## Project Structure

```
Video Editor/
├── main.py              # CLI entry point
├── pipeline.py          # LangGraph graph definition
├── state_types.py       # CriticState, AUDIENCE_OPTIONS, report models
├── server.py             # HTTP server (UI + /api/run + /api/evaluate)
├── ui.html               # React UI (video + report, click-to-seek)
├── requirements.txt
├── .env.example
└── nodes/
    ├── ingest.py        # FFmpeg extraction
    ├── vision.py        # Claude Vision frame analysis (+ authenticity)
    ├── audio.py         # Whisper transcription + silence detection
    └── critic.py        # LLM aggregation + final report
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Anthropic API key |
| `PORT` | 7474 | Server port |
| `CLAUDE_MODEL` | claude-sonnet-4-6 | Model for Vision + Critic |
| `MAX_VISION_FRAMES` | 8 | Max frames sent to Claude Vision |
| `MAX_FRAMES` | 8 | Alias for MAX_VISION_FRAMES |
| `REJECT_THRESHOLD` | 50 | Score below which verdict = reject |
| `REFINE_THRESHOLD` | 80 | Score below which verdict = refine |
| `WHISPER_MODEL` | base | Whisper model size |
| `SILENCE_THRESHOLD_DB` | -35 | Silence detection threshold |
| `MIN_SILENCE_DURATION` | 2.0 | Min silence (seconds) to flag |
