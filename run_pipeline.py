"""
run_pipeline.py — Subprocess entry point for the critic pipeline.

Runs in a clean interpreter so Whisper/numba load without print-monkey-patching conflicts.
Called by server.py via subprocess.

Usage:
  python run_pipeline.py <video_path> <output_dir> <audience>
"""

import sys
from pathlib import Path

# Load .env before importing pipeline
from dotenv import load_dotenv
load_dotenv()

from pipeline import run_critic


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_pipeline.py <video_path> <output_dir> <audience>", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    audience = sys.argv[3]

    try:
        report = run_critic(video_path, output_dir=output_dir, audience=audience)
        # Print report as JSON on a single line for the server to parse
        import json
        print(f"\n__REPORT__{json.dumps(report)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
