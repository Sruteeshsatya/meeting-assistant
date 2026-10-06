"""CLI: python cli.py meeting.mp3"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from meeting_assistant.config import RUNS_DIR, Settings
from meeting_assistant.pipeline import STAGES, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="AI meeting assistant (CLI)")
    parser.add_argument("audio", type=Path, help="Path to the meeting recording")
    parser.add_argument("--context", default="", help="Optional domain terms / names (comma separated)")
    parser.add_argument("--context-file", type=Path, help="Read the context from a text file instead")
    parser.add_argument("--out", type=Path, default=RUNS_DIR, help="Output folder (default: runs/)")
    parser.add_argument("--whisper", help="faster-whisper model (default from WHISPER_MODEL or 'auto')")
    parser.add_argument("--refine-model", help="Gemini model for stage 2")
    parser.add_argument("--docs-model", help="Gemini model for stage 3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = Settings()
    if args.whisper:
        settings.whisper_model = args.whisper
    if args.refine_model:
        settings.refine_model = args.refine_model
    if args.docs_model:
        settings.docs_model = args.docs_model
    context = args.context_file.read_text(encoding="utf-8") if args.context_file else args.context

    if not args.audio.exists():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        return 2

    def make_transcriber():
        from meeting_assistant.transcribe import WhisperTranscriber

        return WhisperTranscriber(settings.whisper_model)

    def make_llm():
        from meeting_assistant.llm import GeminiClient

        return GeminiClient(settings.gemini_api_key or "")

    def print_stage(stage, state):
        print(f"[{state:>7}] {STAGES[stage]}")

    result = run_pipeline(
        args.audio.read_bytes(), args.audio.name, stt=make_transcriber, llm=make_llm, settings=settings,
        context=context, on_stage=print_stage, save_dir=args.out,
    )
    if result.error:
        print(f"\nFailed during '{STAGES[result.failed_stage]}': {result.error.user_message}", file=sys.stderr)
        if result.error.detail:
            print(f"  detail: {result.error.detail}", file=sys.stderr)
        return 1
    record = result.record
    print(f"\n{record.title}\n  decisions: {len(record.decisions)}  action items: {len(record.action_items)}  "
          f"transcript fixes: {sum(c.applied for c in result.corrections)}")
    print(f"Outputs saved to: {result.saved_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
