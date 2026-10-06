"""Fake models for tests."""

from __future__ import annotations

import io
import wave

import numpy as np

from meeting_assistant.schema import (
    DraftActionItem, ProposedFix, DraftDecision, DocumentationResponse, MinutesTopic, DraftOpenItem,
    RefinementResponse, Segment, Transcript,
)

SEGMENTS = [
    "Okay let's start. Today we review the cuber netties migration and the pie torch upgrade.",
    "Priya, can you finish the cuber netties cluster setup by Friday?",
    "Sure, I'll have it done by Friday.",
    "I think we should maybe move to Postgres sixteen, but let's discuss that next week.",
    "We agreed we will not ship the old dashboard in release two point five.",
    "Someone also needs to update the CI pipeline docs.",
]


def make_transcript() -> Transcript:
    segs = [Segment(id=i, start=i * 10.0, end=i * 10.0 + 9, text=t) for i, t in enumerate(SEGMENTS)]
    return Transcript(segments=segs, duration=60.0, model="fake-stt")


def wav_bytes(seconds: float = 2.0, amplitude: float = 0.3, sr: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


class FakeSTT:
    name = "fake-stt"

    def transcribe(self, samples, duration, *, hint="", progress=None):
        if progress:
            progress(1.0, "done")
        return make_transcript()


REFINEMENT = RefinementResponse(
    corrections=[
        ProposedFix(segment_id=0, original="cuber netties", corrected="Kubernetes", reason="misheard tool"),
        ProposedFix(segment_id=0, original="pie torch", corrected="PyTorch", reason="misheard library"),
        ProposedFix(segment_id=1, original="cuber netties", corrected="Kubernetes", reason="misheard tool"),
        # changes a number
        ProposedFix(segment_id=3, original="Postgres sixteen", corrected="Postgres 15", reason="bad"),
        # removes negation
        ProposedFix(segment_id=4, original="will not ship", corrected="will ship", reason="bad"),
        # text not present
        ProposedFix(segment_id=5, original="see eye", corrected="CI", reason="acronym"),
    ]
)

DOCUMENTATION = DocumentationResponse(
    title="Kubernetes migration sync",
    summary="The team reviewed the Kubernetes migration and PyTorch upgrade.",
    participants_mentioned=["Priya", "Rahul"],  # Rahul never mentioned
    minutes=[MinutesTopic(topic="Migration", points=["Cluster setup is in progress."])],
    decisions=[
        DraftDecision(decision="Do not ship the old dashboard in release 2.5.",
                    evidence="We agreed we will not ship the old dashboard in release two point five."),
        DraftDecision(decision="Hire two contractors.", evidence="we will hire two contractors"),  # invented
    ],
    action_items=[
        DraftActionItem(task="Finish the Kubernetes cluster setup", owner="Priya", deadline="by Friday",
                      evidence="Priya, can you finish the Kubernetes cluster setup by Friday?"),
        DraftActionItem(task="Update the CI pipeline docs", owner="Rahul", deadline="Monday",  # invented
                      evidence="Someone also needs to update the CI pipeline docs."),
        DraftActionItem(task="Review the PyTorch upgrade", owner="TBD", deadline=None,
                      evidence="the PyTorch upgrade"),
    ],
    open_items=[DraftOpenItem(item="Move to Postgres 16", evidence="I think we should maybe move to Postgres sixteen")],
)


class FakeLLM:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def generate(self, *, model, system, prompt, schema):
        self.calls.append((model, schema.__name__))
        if self.fail_on == schema.__name__:
            from meeting_assistant.errors import LLMError

            raise LLMError("Gemini rate limit or quota exceeded.")
        if schema is RefinementResponse:
            return REFINEMENT
        if schema is DocumentationResponse:
            return DOCUMENTATION
        raise AssertionError(schema)
