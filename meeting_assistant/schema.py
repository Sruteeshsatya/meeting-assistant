"""Data models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

UNSPECIFIED = "Unspecified"


# Transcripts
class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    segments: list[Segment]
    duration: float = 0.0
    language: str = "en"
    model: str = ""

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def as_timestamped_text(self) -> str:
        return "\n".join(f"[{format_time(s.start)}] {s.text.strip()}" for s in self.segments)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


# Refinement (LLM 1)
# descriptions guide Gemini
class ProposedFix(BaseModel):
    segment_id: int = Field(description="ID of the segment that contains the error.")
    original: str = Field(description="The misrecognised text, copied EXACTLY from the segment.")
    corrected: str = Field(description="The corrected replacement text.")
    reason: str = Field(description="Short reason, e.g. 'acronym misheard as words'.")


class RefinementResponse(BaseModel):
    corrections: list[ProposedFix]


class Correction(BaseModel):
    segment_id: int
    timestamp: str
    original: str
    corrected: str
    reason: str
    applied: bool
    rejected_because: Optional[str] = None


# Documentation (LLM 2)
class MinutesTopic(BaseModel):
    topic: str = Field(description="Short topic heading.")
    points: list[str] = Field(description="Concise bullet points of what was discussed, in order.")


class DraftDecision(BaseModel):
    decision: str = Field(description="The decision that was explicitly agreed.")
    evidence: str = Field(description="Short verbatim excerpt from the transcript showing the agreement.")


class DraftActionItem(BaseModel):
    task: str = Field(description="The work to be done, phrased as an action.")
    owner: Optional[str] = Field(description="Person/team explicitly named as responsible, else null.")
    deadline: Optional[str] = Field(description="Deadline exactly as stated in the meeting, else null.")
    evidence: str = Field(description="Short verbatim excerpt from the transcript showing the task.")


class DraftOpenItem(BaseModel):
    item: str = Field(description="Proposal, suggestion or question raised but NOT agreed.")
    evidence: str = Field(description="Short verbatim excerpt from the transcript.")


class DocumentationResponse(BaseModel):
    title: str = Field(description="A short descriptive meeting title based on its content.")
    summary: str = Field(description="3-5 sentence summary of the meeting.")
    participants_mentioned: list[str] = Field(description="Names explicitly spoken in the meeting.")
    minutes: list[MinutesTopic]
    decisions: list[DraftDecision]
    action_items: list[DraftActionItem]
    open_items: list[DraftOpenItem]


# Final record
class Decision(BaseModel):
    id: str
    decision: str
    evidence: str
    timestamp: Optional[str] = None
    evidence_verified: bool = True


class ActionItem(BaseModel):
    id: str
    task: str
    owner: str = UNSPECIFIED
    deadline: str = UNSPECIFIED
    evidence: str
    timestamp: Optional[str] = None
    evidence_verified: bool = True
    notes: list[str] = Field(default_factory=list)


class OpenItem(BaseModel):
    item: str
    evidence: str
    timestamp: Optional[str] = None


class RecordMetadata(BaseModel):
    source_file: str
    duration: str
    generated_at: str
    speech_to_text_model: str
    refinement_model: str
    documentation_model: str


class MeetingRecord(BaseModel):
    metadata: RecordMetadata
    title: str
    summary: str
    participants_mentioned: list[str]
    minutes: list[MinutesTopic]
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_items: list[OpenItem]
    validation_notes: list[str] = Field(default_factory=list)
