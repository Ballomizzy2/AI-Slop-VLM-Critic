"""
main.py — CLI entry point for the critic pipeline.

Usage:
  python main.py <video_path>
  python main.py video.mp4
  python main.py video.mp4 --output ./results
"""

import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

from pipeline import run_critic


def print_report(report: dict):
    """Pretty-print the report to terminal."""
    verdict = report.get("verdict", "unknown").upper()
    score = report.get("score", 0)
    auth = report.get("authenticity_score")
    audience = report.get("audience_fit_score")
    issues = report.get("issues", [])
    summary = report.get("summary", "")

    verdict_emoji = {"PASS": "[OK]", "REFINE": "[~]", "REJECT": "[X]"}.get(verdict, "[?]")

    print(f"\n{verdict_emoji}  VERDICT: {verdict}  |  SCORE: {score}/100")
    if auth is not None:
        print(f"   AUTHENTICITY: {auth}/100")
    if audience is not None:
        print(f"   AUDIENCE FIT: {audience}/100")
    print(f"{summary}\n")

    if issues:
        print(f"[!] Issues Found ({len(issues)}):")
        print("-" * 50)
        for i, issue in enumerate(issues, 1):
            sev = issue.get("severity", "?").upper()
            ts = issue.get("timestamp", "?")
            itype = issue.get("type", "?")
            desc = issue.get("description", "")
            rec = issue.get("recommendation", "")
            print(f"\n  {i}. [{sev}] {itype} @ {ts}")
            print(f"     Problem: {desc}")
            print(f"     Fix:     {rec}")
    else:
        print("[OK] No issues found!")

    retry = report.get("retry", False)
    if retry:
        print(f"\n[~] Retry recommended: Yes")


def main():
    parser = argparse.ArgumentParser(
        description="Critic Pipeline — AI-powered video quality reviewer"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--audience", default="general", help="Target audience (default: general)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON report")
    args = parser.parse_args()

    try:
        report = run_critic(args.video, output_dir=args.output, audience=args.audience)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_report(report)

    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[X] Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
