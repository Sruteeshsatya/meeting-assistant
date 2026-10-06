"""Export files."""

from __future__ import annotations

import io
import json
import zipfile

from .schema import UNSPECIFIED, Correction, MeetingRecord, Transcript


def escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def record_to_markdown(record: MeetingRecord) -> str:
    # same data as JSON
    meta = record.metadata
    lines = [
        f"# {record.title}",
        "",
        f"- **Source recording:** {meta.source_file}",
        f"- **Duration:** {meta.duration}",
        f"- **Generated:** {meta.generated_at}",
        f"- **Models:** speech-to-text `{meta.speech_to_text_model}` · refinement `{meta.refinement_model}` · "
        f"documentation `{meta.documentation_model}`",
    ]
    if record.participants_mentioned:
        lines.append(f"- **Names mentioned:** {', '.join(record.participants_mentioned)}")
    lines += ["", "## Summary", "", record.summary, "", "## Minutes", ""]
    if record.minutes:
        for topic in record.minutes:
            lines.append(f"### {topic.topic}")
            lines += [f"- {point}" for point in topic.points]
            lines.append("")
    else:
        lines += ["_No minutes were generated._", ""]

    lines += ["## Key decisions", ""]
    if record.decisions:
        for decision in record.decisions:
            warning = "" if decision.evidence_verified else " ⚠️ _quote not matched - verify_"
            time_label = f" [{decision.timestamp}]" if decision.timestamp else ""
            lines.append(f"{decision.id}. **{decision.decision}**{warning}  ")
            lines.append(f"   > “{decision.evidence}”{time_label}")
    else:
        lines.append("_No decisions were reached in this meeting._")
    lines.append("")

    lines += ["## Action items", ""]
    if record.action_items:
        lines += ["| # | Task | Owner | Deadline | Evidence |", "|---|---|---|---|---|"]
        for item in record.action_items:
            owner = f"_{UNSPECIFIED}_" if item.owner == UNSPECIFIED else escape_cell(item.owner)
            deadline = f"_{UNSPECIFIED}_" if item.deadline == UNSPECIFIED else escape_cell(item.deadline)
            time_label = f" [{item.timestamp}]" if item.timestamp else ""
            lines.append(f"| {item.id} | {escape_cell(item.task)} | {owner} | {deadline} | “{escape_cell(item.evidence)}”{time_label} |")
    else:
        lines.append("_No action items were agreed in this meeting._")
    lines.append("")

    if record.open_items:
        lines += ["## Open items (raised but not agreed)", ""]
        lines += [f"- {o.item}" + (f" [{o.timestamp}]" if o.timestamp else "") for o in record.open_items]
        lines.append("")
    if record.validation_notes:
        lines += ["## Validation notes", ""]
        lines += [f"- {note}" for note in record.validation_notes]
        lines.append("")
    return "\n".join(lines)


def record_to_json(record: MeetingRecord) -> str:
    return record.model_dump_json(indent=2)


def transcript_to_text(transcript: Transcript, title: str) -> str:
    return f"{title}\nModel: {transcript.model}\n\n{transcript.as_timestamped_text()}\n"


def corrections_to_json(corrections: list[Correction]) -> str:
    return json.dumps([c.model_dump() for c in corrections], indent=2, ensure_ascii=False)


def build_outputs(raw: Transcript, refined: Transcript, corrections: list[Correction], record: MeetingRecord) -> dict[str, str]:
    return {
        "raw_transcript.txt": transcript_to_text(raw, "RAW TRANSCRIPT (speech-to-text output, before refinement)"),
        "refined_transcript.txt": transcript_to_text(refined, "REFINED TRANSCRIPT (after domain-term correction)"),
        "refinement_changes.json": corrections_to_json(corrections),
        "meeting_record.md": record_to_markdown(record),
        "meeting_record.json": record_to_json(record),
    }


def build_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
