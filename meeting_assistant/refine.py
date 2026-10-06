"""Stage 2: transcript refinement (LLM 1)."""

from __future__ import annotations

import re
from typing import Callable, Optional

from .config import PROMPTS_DIR
from .errors import LLMError, RefinementError
from .guardrails import check_correction
from .llm import LLMClient
from .schema import Correction, ProposedFix, RefinementResponse, Segment, Transcript, format_time

ProgressCallback = Callable[[float, str], None]


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def split_into_chunks(segments: list[Segment], max_words: int) -> list[list[Segment]]:
    chunks: list[list[Segment]] = [[]]
    word_count = 0
    for segment in segments:
        words = len(segment.text.split())
        if chunks[-1] and word_count + words > max_words:
            chunks.append([])
            word_count = 0
        chunks[-1].append(segment)
        word_count += words
    return [chunk for chunk in chunks if chunk]


def build_prompt(chunk: list[Segment], context: str, part: int, total_parts: int) -> str:
    context = context.strip() or "(none provided)"
    lines = "\n".join(f"[{s.id}] ({format_time(s.start)}) {s.text}" for s in chunk)
    part_label = f" (part {part} of {total_parts})" if total_parts > 1 else ""
    return (
        f"MEETING CONTEXT / GLOSSARY FROM THE USER:\n{context}\n\n"
        f"ASR TRANSCRIPT SEGMENTS{part_label}:\n{lines}\n\n"
        "List the corrections (JSON only)."
    )


def replace_text(text: str, original: str, corrected: str) -> Optional[str]:
    if original in text:
        return text.replace(original, corrected)
    pattern = re.compile(re.escape(original), re.I)  # try ignoring case
    if pattern.search(text):
        return pattern.sub(lambda _match: corrected, text)
    return None


def apply_corrections(raw: Transcript, fixes: list[ProposedFix]) -> tuple[Transcript, list[Correction]]:
    """Apply safe fixes, log all."""
    segments_by_id = {s.id: s.model_copy() for s in raw.segments}  # raw stays untouched
    changes: list[Correction] = []
    already_seen: set[tuple[int, str, str]] = set()
    for fix in fixes:
        key = (fix.segment_id, fix.original, fix.corrected)
        if key in already_seen:
            continue
        already_seen.add(key)
        segment = segments_by_id.get(fix.segment_id)
        timestamp = format_time(segment.start) if segment else "--:--"
        problem = None if segment else "segment id does not exist"
        if problem is None:
            problem = check_correction(fix.original, fix.corrected)
        new_text = None
        if problem is None:
            new_text = replace_text(segment.text, fix.original, fix.corrected)
            if new_text is None:
                problem = "original text not found in that segment"
        if problem is None:
            segment.text = new_text
        changes.append(
            Correction(
                segment_id=fix.segment_id, timestamp=timestamp, original=fix.original, corrected=fix.corrected,
                reason=fix.reason, applied=problem is None, rejected_because=problem,
            )
        )
    refined = raw.model_copy(update={"segments": [segments_by_id[s.id] for s in raw.segments]})
    return refined, changes


def refine_transcript(
    raw: Transcript,
    llm: LLMClient,
    *,
    model: str,
    context: str = "",
    words_per_chunk: int = 3000,
    progress: Optional[ProgressCallback] = None,
) -> tuple[Transcript, list[Correction]]:
    system_prompt = load_prompt("refine_system.md")
    chunks = split_into_chunks(raw.segments, words_per_chunk)
    fixes: list[ProposedFix] = []
    for number, chunk in enumerate(chunks, start=1):
        if progress:
            progress((number - 1) / len(chunks), f"Refining part {number} of {len(chunks)}")
        chunk_ids = {s.id for s in chunk}
        try:
            reply = llm.generate(
                model=model, system=system_prompt, prompt=build_prompt(chunk, context, number, len(chunks)),
                schema=RefinementResponse,
            )
        except LLMError as exc:
            raise RefinementError(exc.user_message, detail=exc.detail) from exc
        for fix in reply.corrections:
            # wrong chunk: reject
            fixes.append(fix if fix.segment_id in chunk_ids else fix.model_copy(update={"segment_id": -1}))
    if progress:
        progress(1.0, "Refinement complete")
    return apply_corrections(raw, fixes)
