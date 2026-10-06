"""Transcriber with a fake Whisper model."""

from types import SimpleNamespace

import numpy as np
import pytest

from meeting_assistant import transcribe as T
from meeting_assistant.errors import TranscriptionError


class FakeModel:
    def __init__(self, name, device, fail=False, empty=False):
        self.name, self.device, self.fail, self.empty = name, device, fail, empty

    def transcribe(self, samples, **kw):
        def gen():
            if self.fail:
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            if not self.empty:
                yield SimpleNamespace(start=0.0, end=2.0, text=" Hello team. ")
                yield SimpleNamespace(start=2.0, end=4.0, text="   ")
                yield SimpleNamespace(start=4.0, end=6.0, text="Let's ship on Friday.")
        return gen(), SimpleNamespace(language="en")


def make_transcriber(monkeypatch, device, gpu_fails=False, cpu_fails=False, empty=False):
    monkeypatch.setattr(T, "pick_device", lambda: (device, "float16" if device == "cuda" else "int8"))
    monkeypatch.setattr(T, "load_gpu_libraries", lambda: None)
    loads = []

    def fake_load(name, device_name, compute_type):
        loads.append((name, device_name))
        fail = (device_name == "cuda" and gpu_fails) or (device_name == "cpu" and cpu_fails)
        return FakeModel(name, device_name, fail=fail, empty=empty)

    monkeypatch.setattr(T.WhisperTranscriber, "load_model", staticmethod(fake_load))
    return T.WhisperTranscriber("auto"), loads


AUDIO = np.zeros(16000 * 6, np.float32)


def test_gpu_success(monkeypatch):
    stt, loads = make_transcriber(monkeypatch, "cuda")
    t = stt.transcribe(AUDIO, 6.0)
    assert loads == [("large-v3", "cuda")]
    assert [s.text for s in t.segments] == ["Hello team.", "Let's ship on Friday."]
    assert [s.id for s in t.segments] == [0, 1] and t.model == "faster-whisper large-v3 (cuda)"


def test_gpu_runtime_failure_falls_back_to_cpu(monkeypatch):
    stt, loads = make_transcriber(monkeypatch, "cuda", gpu_fails=True)
    msgs = []
    t = stt.transcribe(AUDIO, 6.0, progress=lambda f, m: msgs.append(m))
    assert loads == [("large-v3", "cuda"), ("small.en", "cpu")]
    assert len(t.segments) == 2 and "cpu fallback" in t.model
    assert any("retrying on CPU" in m for m in msgs)


def test_both_fail_gives_clear_error(monkeypatch):
    stt, _ = make_transcriber(monkeypatch, "cuda", gpu_fails=True, cpu_fails=True)
    with pytest.raises(TranscriptionError, match="both the GPU and the CPU") as e:
        stt.transcribe(AUDIO, 6.0)
    assert "libcublas" in e.value.detail


def test_cpu_failure_is_reported(monkeypatch):
    stt, _ = make_transcriber(monkeypatch, "cpu", cpu_fails=True)
    assert stt.model_name == "small.en"
    with pytest.raises(TranscriptionError, match="failed while processing"):
        stt.transcribe(AUDIO, 6.0)


def test_no_speech(monkeypatch):
    stt, _ = make_transcriber(monkeypatch, "cuda", empty=True)
    with pytest.raises(TranscriptionError, match="No speech"):
        stt.transcribe(AUDIO, 6.0)


def test_preload_is_safe_without_nvidia_packages():
    T.load_gpu_libraries()  # no nvidia packages here
