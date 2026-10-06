from meeting_assistant.guardrails import (
    check_correction, count_negations, find_numbers, is_empty_value, find_quote, appears_in_transcript,
)
from tests.fakes import make_transcript


def test_numbers_words_and_digits_match():
    assert find_numbers("twenty five users") == find_numbers("25 users")
    assert find_numbers("GPT four") == find_numbers("GPT-4")
    assert find_numbers("1,500 requests") == find_numbers("one thousand five hundred requests")
    assert find_numbers("version 3.5") != find_numbers("version 3.6")


def test_negation_count():
    assert count_negations("we won't ship it") == 1
    assert count_negations("we will not ship, never") == 2
    assert count_negations("we will ship") == 0


def test_check_correction():
    assert check_correction("cuber netties", "Kubernetes") is None
    assert check_correction("see eye see dee", "CI/CD") is None
    assert check_correction("Postgres sixteen", "Postgres 15") == "would change a number"
    assert check_correction("will not ship", "will ship") == "would change negation"
    assert check_correction("same", "same") == "no change"
    assert "too long" in check_correction(" ".join(["word"] * 13), "x")


def test_find_quote():
    transcript = make_transcript()
    found, timestamp = find_quote("Sure, I'll have it done by Friday", transcript.segments)
    assert found and timestamp == "00:20"
    found, _ = find_quote("we will hire two contractors", transcript.segments)
    assert not found


def test_owner_and_deadline_support():
    text = make_transcript().text
    assert appears_in_transcript("Priya", text)
    assert not appears_in_transcript("Rahul", text)
    assert appears_in_transcript("by Friday", text)
    assert not appears_in_transcript("October 9", text)
    assert is_empty_value("TBD") and is_empty_value(None) and is_empty_value(" n/a ")
    assert not is_empty_value("Priya")
