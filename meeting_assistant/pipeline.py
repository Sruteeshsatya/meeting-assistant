"""Runs the three stages."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .audio import load_audio
from .config import Settings
from .document import document_meeting
from .errors import PipelineError
from .llm import LLMClient
from .refine import refine_transcript
from .render import build_outputs
from .schema import Correction, MeetingRecord, Transcript
from .transcribe import SpeechToText

log = logging.getLogger(__name__)

STAGES = {
    "validate": "Checking the audio file",
    "transcribe": "Stage 1 · Speech-to-text",
    "refine": "Stage 2 · Transcript refinement",
    "document": "Stage 3 · Minutes, decisions & action items",
    "export": "Preparing downloads",
}

StageCallback = Callable[[str, str], None]  # (stage, "running" | "done" | "failed")
ProgressCallback = Callable[[str, float, str], None]  # (stage, fraction, message)


@dataclass
class PipelineResult:
    filename: str
    raw: Optional[Transcript] = None
    refined: Optional[Transcript] = None
    corrections: list[Correction] = field(default_factory=list)
    record: Optional[MeetingRecord] = None
    outputs: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[PipelineError] = None
    failed_stage: Optional[str] = None
    saved_to: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.record is not None


def run_pipeline(
    data: bytes,
    filename: str,
    *,
    stt: SpeechToText | Callable[[], SpeechToText],
    llm: LLMClient | Callable[[], LLMClient],
    settings: Settings,
    context: str = "",
    on_stage: Optional[StageCallback] = None,
    on_progress: Optional[ProgressCallback] = None,
    save_dir: Optional[Path] = None,
) -> PipelineResult:
    """Run all stages in order."""
    # models load lazily
    result = PipelineResult(filename=filename)
    report_stage = on_stage or (lambda *_: None)
    report_progress = on_progress or (lambda *_: None)
    current_stage = "validate"

    def start_stage(stage: str) -> float:
        nonlocal current_stage
        current_stage = stage
        report_stage(stage, "running")
        return time.perf_counter()

    def finish_stage(stage: str, started_at: float) -> None:
        result.timings[stage] = round(time.perf_counter() - started_at, 2)
        report_stage(stage, "done")

    try:
        started = start_stage("validate")
        samples, duration = load_audio(data, filename)
        finish_stage("validate", started)

        started = start_stage("transcribe")
        speech_model = stt() if callable(stt) and not hasattr(stt, "transcribe") else stt
        hint = context if settings.use_glossary_hint else ""
        result.raw = speech_model.transcribe(
            samples, duration, hint=hint, progress=lambda f, m: report_progress("transcribe", f, m)
        )
        finish_stage("transcribe", started)

        started = start_stage("refine")
        language_model = llm() if callable(llm) and not hasattr(llm, "generate") else llm
        result.refined, result.corrections = refine_transcript(
            result.raw, language_model, model=settings.refine_model, context=context,
            words_per_chunk=settings.words_per_chunk, progress=lambda f, m: report_progress("refine", f, m),
        )
        finish_stage("refine", started)

        started = start_stage("document")
        result.record = document_meeting(
            result.refined, language_model, model=settings.docs_model, context=context, source_file=filename,
            stt_model=result.raw.model, refine_model=settings.refine_model,
        )
        finish_stage("document", started)

        started = start_stage("export")
        result.outputs = build_outputs(result.raw, result.refined, result.corrections, result.record)
        if save_dir is not None:
            result.saved_to = save_outputs(result.outputs, save_dir, filename)
        finish_stage("export", started)
    except PipelineError as exc:
        result.error, result.failed_stage = exc, current_stage
        report_stage(current_stage, "failed")
        log.warning("Pipeline failed at %s: %s (%s)", current_stage, exc.user_message, exc.detail)
    except Exception as exc:  # unexpected bug
        log.exception("Unexpected pipeline error")
        result.error = PipelineError(
            f"An unexpected error occurred during '{STAGES[current_stage]}'.", detail=f"{type(exc).__name__}: {exc}"
        )
        result.failed_stage = current_stage
        report_stage(current_stage, "failed")
    return result


def save_outputs(outputs: dict[str, str], base_dir: Path, filename: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem)[:40] or "meeting"
    folder = base_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{safe_name}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (folder / name).write_text(content, encoding="utf-8")
    return folder
