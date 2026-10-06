"""Gemini client with a fake SDK."""

from types import SimpleNamespace

import pytest

from meeting_assistant import llm as llm_mod
from meeting_assistant.errors import LLMError
from meeting_assistant.llm import GeminiClient, parse_json_response
from meeting_assistant.schema import RefinementResponse


class APIErr(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(parsed=None, text=item, candidates=[SimpleNamespace(finish_reason="STOP")])


def make_client(script):
    c = GeminiClient.__new__(GeminiClient)
    c.max_retries = 5
    c._client = SimpleNamespace(models=FakeModels(script))
    return c


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


GOOD = '{"corrections": [{"segment_id": 1, "original": "pie torch", "corrected": "PyTorch", "reason": "lib"}]}'


def test_parses_fenced_json():
    r = parse_json_response("```json\n" + GOOD + "\n```", RefinementResponse)
    assert r.corrections[0].corrected == "PyTorch"


def test_retries_rate_limit_then_succeeds():
    c = make_client([APIErr(429), APIErr(503), GOOD])
    r = c.generate(model="m", system="s", prompt="p", schema=RefinementResponse)
    assert r.corrections[0].segment_id == 1 and c._client.models.calls == 3


def test_retries_invalid_json():
    c = make_client(["not json", GOOD])
    assert c.generate(model="m", system="s", prompt="p", schema=RefinementResponse).corrections


def test_bad_key_message():
    c = make_client([APIErr(403)])
    with pytest.raises(LLMError, match="API key was rejected"):
        c.generate(model="m", system="s", prompt="p", schema=RefinementResponse)


def test_unknown_model_message():
    c = make_client([APIErr(404)])
    with pytest.raises(LLMError, match="not found"):
        c.generate(model="gemini-nope", system="s", prompt="p", schema=RefinementResponse)


def test_gives_up_after_repeated_bad_output():
    c = make_client([""] * 5)
    with pytest.raises(LLMError, match="did not return a usable result"):
        c.generate(model="m", system="s", prompt="p", schema=RefinementResponse)


def test_missing_key():
    with pytest.raises(LLMError, match="No Gemini API key"):
        GeminiClient("")
