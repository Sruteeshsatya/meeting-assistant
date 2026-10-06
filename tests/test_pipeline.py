import json

from meeting_assistant.config import Settings
from meeting_assistant.document import document_meeting
from meeting_assistant.pipeline import run_pipeline
from meeting_assistant.refine import split_into_chunks, refine_transcript
from meeting_assistant.schema import UNSPECIFIED
from tests.fakes import FakeLLM, FakeSTT, make_transcript, wav_bytes


def test_refinement_applies_safe_edits_only():
    raw = make_transcript()
    refined, changes = refine_transcript(raw, FakeLLM(), model="m")
    assert "Kubernetes" in refined.segments[0].text and "PyTorch" in refined.segments[0].text
    assert "Kubernetes" in refined.segments[1].text
    assert refined.segments[3].text == raw.segments[3].text  # number kept
    assert "will not ship" in refined.segments[4].text  # negation kept
    status = {c.original: c.rejected_because for c in changes}
    assert status["Postgres sixteen"] == "would change a number"
    assert status["will not ship"] == "would change negation"
    assert status["see eye"] == "original text not found in that segment"
    assert raw.segments[0].text.startswith("Okay let's start. Today we review the cuber")  # raw untouched


def test_chunking_keeps_all_segments():
    segments = make_transcript().segments
    chunks = split_into_chunks(segments, max_words=20)
    assert len(chunks) > 1
    assert [s.id for chunk in chunks for s in chunk] == [s.id for s in segments]


def test_documentation_checks():
    refined, _ = refine_transcript(make_transcript(), FakeLLM(), model="m")
    record = document_meeting(refined, FakeLLM(), model="docs", source_file="x.wav", stt_model="s", refine_model="r")
    first, second, third = record.action_items
    assert (first.owner, first.deadline) == ("Priya", "by Friday") and first.evidence_verified
    assert (second.owner, second.deadline) == (UNSPECIFIED, UNSPECIFIED)  # invented values removed
    assert third.owner == UNSPECIFIED and third.deadline == UNSPECIFIED  # "TBD" / null
    real, invented = record.decisions
    assert real.evidence_verified and real.timestamp == "00:40"
    assert not invented.evidence_verified  # invented decision flagged
    assert record.participants_mentioned == ["Priya"]
    assert any("Rahul" in n for n in record.validation_notes)


def test_end_to_end_pipeline_and_consistent_exports(tmp_path):
    stages = []
    result = run_pipeline(
        wav_bytes(), "meeting.wav", stt=FakeSTT(), llm=FakeLLM(), settings=Settings(gemini_api_key="x"),
        on_stage=lambda stage, state: stages.append((stage, state)), save_dir=tmp_path,
    )
    assert result.ok, result.error
    assert [stage for stage, state in stages if state == "done"] == ["validate", "transcribe", "refine", "document", "export"]
    assert set(result.outputs) == {
        "raw_transcript.txt", "refined_transcript.txt", "refinement_changes.json",
        "meeting_record.md", "meeting_record.json",
    }
    assert "cuber netties" in result.outputs["raw_transcript.txt"]
    assert "Kubernetes" in result.outputs["refined_transcript.txt"]
    data = json.loads(result.outputs["meeting_record.json"])
    md = result.outputs["meeting_record.md"]
    for d in data["decisions"]:
        assert d["decision"] in md
    for a in data["action_items"]:
        assert a["task"] in md
    assert "_Unspecified_" in md
    assert (result.saved_to / "meeting_record.json").exists()


def test_two_llm_stages_are_distinct_calls():
    llm = FakeLLM()
    s = Settings(gemini_api_key="x", refine_model="refiner", docs_model="documenter")
    run_pipeline(wav_bytes(), "m.wav", stt=FakeSTT(), llm=llm, settings=s)
    assert llm.calls == [("refiner", "RefinementResponse"), ("documenter", "DocumentationResponse")]


def test_failure_keeps_partial_results():
    result = run_pipeline(wav_bytes(), "m.wav", stt=FakeSTT(), llm=FakeLLM(fail_on="DocumentationResponse"),
                       settings=Settings(gemini_api_key="x"))
    assert not result.ok and result.failed_stage == "document"
    assert result.raw is not None and result.refined is not None and result.record is None
    assert "rate limit" in result.error.user_message


def test_bad_file_fails_at_validation():
    result = run_pipeline(b"", "m.mp3", stt=FakeSTT(), llm=FakeLLM(), settings=Settings(gemini_api_key="x"))
    assert result.failed_stage == "validate" and "empty" in result.error.user_message
