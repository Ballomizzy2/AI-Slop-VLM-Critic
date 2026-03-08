"""
pipeline.py — LangGraph critic pipeline.

Graph: ingest_node → vision_node → audio_node → critic_node

Usage:
  report = run_critic("video.mp4", audience="casual")
"""

import json
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from nodes.ingest import ingest_node
from nodes.vision import vision_node
from nodes.audio  import audio_node
from nodes.critic import critic_node


class CriticState(TypedDict):
    video_path:   str
    audience:     str
    frames:       list
    audio_path:   Optional[str]
    metadata:     dict
    scene_cuts:   list
    frame_scores: list
    transcript:   Optional[str]
    report:       Optional[dict]


def build_pipeline():
    graph = StateGraph(CriticState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("vision", vision_node)
    graph.add_node("audio",  audio_node)
    graph.add_node("critic", critic_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "vision")
    graph.add_edge("vision", "audio")
    graph.add_edge("audio",  "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def run_critic(video_path: str, output_dir: str = "output", audience: str = "general") -> dict:
    """
    Run the full critic pipeline on a video file.

    Args:
        video_path:  Path to video file
        output_dir:  Where to save frames, audio, and report
        audience:    Target audience — general | technical | buyer | casual | educational

    Returns:
        report dict with verdict, scores, and issues
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/frames", exist_ok=True)
    os.makedirs(f"{output_dir}/audio",  exist_ok=True)

    print("\n[Critic Pipeline]")
    print(f"   Video:    {os.path.basename(video_path)}")
    print(f"   Audience: {audience}\n")

    pipeline = build_pipeline()

    final_state = pipeline.invoke({
        "video_path":   video_path,
        "audience":     audience,
        "frames":       [],
        "audio_path":   None,
        "metadata":     {},
        "scene_cuts":   [],
        "frame_scores": [],
        "transcript":   None,
        "report":       None,
    })

    report = final_state.get("report", {})

    base        = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(output_dir, f"{base}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}")

    return report