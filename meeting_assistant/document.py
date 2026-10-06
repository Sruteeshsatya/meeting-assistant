"""Stage 3: meeting documentation (LLM 2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .errors import DocumentationError, LLMError
from .guardrails import appears_in_transcript, find_quote, is_empty_value
from .llm import LLMClient
from .refine import load_prompt
from .schema import (
    UNSPECIFIED, ActionItem, Decision, DocumentationResponse, MeetingRecord, OpenItem, RecordMetadata,
    Transcript, format_time,
)


def build_prompt(refined: Transcript, context: str) -> str:
    context = context.strip() or "(none provided)"
    return (
        "OPTIONAL CONTEXT FROM THE USER (use only to understand terms; never add facts from it):\n"
        f"{context}\n\n"
        f"REFINED MEETING TRANSCRIPT (duration {format_time(refined.duration)}):\n"
        f"{refined.as_timestamped_text()}\n\n"
        "Produce the meeting record as JSON."
    )


def verify_record(draft: DocumentationResponse, refined: Transcript):
    """Check every item against the transcript."""
    segments, transcript_text = refined.segments, refined.text
    notes: list[str] = []

    decisions = []
    for number, draft_decision in enumerate(draft.decisions, start=1):
        found, timestamp = find_quote(draft_decision.evidence, segments)
        if not found:
            notes.append(f"D{number}: supporting quote could not be matched to the transcript - please verify.")
        decisions.append(
            Decision(id=f"D{number}", decision=draft_decision.decision, evidence=draft_decision.evidence,
                     timestamp=timestamp, evidence_verified=found)
        )

    action_items = []
    for number, draft_item in enumerate(draft.action_items, start=1):
        found, timestamp = find_quote(draft_item.evidence, segments)
        item_notes = []
        owner, deadline = UNSPECIFIED, UNSPECIFIED
        if not is_empty_value(draft_item.owner):
            if appears_in_transcript(draft_item.owner, transcript_text):
                owner = draft_item.owner.strip()
            else:
                item_notes.append(f"Owner '{draft_item.owner}' was not found in the transcript, so it is shown as unspecified.")
        if not is_empty_value(draft_item.deadline):
            if appears_in_transcript(draft_item.deadline, transcript_text):
                deadline = draft_item.deadline.strip()
            else:
                item_notes.append(f"Deadline '{draft_item.deadline}' was not found in the transcript, so it is shown as unspecified.")
        if not found:
            item_notes.append("Supporting quote could not be matched to the transcript - please verify.")
        notes.extend(f"A{number}: {note}" for note in item_notes)
        action_items.append(
            ActionItem(id=f"A{number}", task=draft_item.task, owner=owner, deadline=deadline,
                       evidence=draft_item.evidence, timestamp=timestamp, evidence_verified=found, notes=item_notes)
        )

    open_items = []
    for draft_open in draft.open_items:
        _, timestamp = find_quote(draft_open.evidence, segments)
        open_items.append(OpenItem(item=draft_open.item, evidence=draft_open.evidence, timestamp=timestamp))

    participants = [
        name for name in draft.participants_mentioned if name.strip() and appears_in_transcript(name, transcript_text)
    ]
    return decisions, action_items, open_items, participants, notes


def document_meeting(
    refined: Transcript,
    llm: LLMClient,
    *,
    model: str,
    context: str = "",
    source_file: str = "",
    stt_model: str = "",
    refine_model: str = "",
    now: Optional[datetime] = None,
) -> MeetingRecord:
    try:
        draft = llm.generate(
            model=model, system=load_prompt("document_system.md"), prompt=build_prompt(refined, context),
            schema=DocumentationResponse,
        )
    except LLMError as exc:
        raise DocumentationError(exc.user_message, detail=exc.detail) from exc

    decisions, action_items, open_items, participants, notes = verify_record(draft, refined)
    now = now or datetime.now(timezone.utc)
    return MeetingRecord(
        metadata=RecordMetadata(
            source_file=source_file, duration=format_time(refined.duration),
            generated_at=now.isoformat(timespec="seconds"), speech_to_text_model=stt_model,
            refinement_model=refine_model, documentation_model=model,
        ),
        title=draft.title.strip() or "Meeting record",
        summary=draft.summary.strip(),
        participants_mentioned=participants,
        minutes=draft.minutes,
        decisions=decisions,
        action_items=action_items,
        open_items=open_items,
        validation_notes=notes,
    )
