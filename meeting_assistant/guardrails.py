"""Checks on model output."""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from .schema import Segment, format_time

# Number words
SMALL_NUMBERS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
TENS = {
    word: 10 * value
    for value, word in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())
    if word != "_"
}
MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
NUMBER_WORDS = set(SMALL_NUMBERS) | set(TENS) | set(MULTIPLIERS)


def words_to_number(words: list[str]) -> int:
    total, current = 0, 0
    for word in words:
        if word in SMALL_NUMBERS:
            current += SMALL_NUMBERS[word]
        elif word in TENS:
            current += TENS[word]
        elif word == "hundred":
            current = max(current, 1) * 100
        elif word in MULTIPLIERS:
            total += max(current, 1) * MULTIPLIERS[word]
            current = 0
    return total + current


def find_numbers(text: str) -> Counter:
    """All numbers in text, digits or words."""
    numbers: list[str] = []
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower().replace("-", " "))
    number_words: list[str] = []

    def save_number_words():
        if number_words:
            numbers.append(str(words_to_number(number_words)))
            number_words.clear()

    for token in tokens:
        if token[0].isdigit():
            save_number_words()
            digits = token.replace(",", "")
            numbers.append(str(float(digits)).rstrip("0").rstrip(".") if "." in digits else str(int(digits)))
        elif token in NUMBER_WORDS:
            number_words.append(token)
        elif token == "and" and number_words:
            continue  # "one hundred and five"
        else:
            save_number_words()
    save_number_words()
    return Counter(numbers)


# Negation
NEGATION_PATTERN = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without)\b|n't\b", re.I)


def count_negations(text: str) -> int:
    return len(NEGATION_PATTERN.findall(text.replace("’", "'")))


def check_correction(original: str, corrected: str) -> str | None:
    """Reason to reject, or None."""
    if not original.strip() or not corrected.strip():
        return "empty original or replacement"
    if original.strip() == corrected.strip():
        return "no change"
    if find_numbers(original) != find_numbers(corrected):
        return "would change a number"
    if count_negations(original) != count_negations(corrected):
        return "would change negation"
    original_words, corrected_words = len(original.split()), len(corrected.split())
    if original_words > 12:
        return "edit too long (refinement must be a local term fix, not a rewrite)"
    if corrected_words > original_words + 3:
        return "replacement adds too many words"
    return None


# Quote matching
def clean_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_quote(quote: str, segments: list[Segment], min_score: float = 80.0) -> tuple[bool, str | None]:
    """Return (found, timestamp)."""
    target = clean_text(quote)
    if not target or not segments:
        return False, None
    best_score, best_index = (0.0, 0.0), None
    for index in range(len(segments)):
        nearby_text = clean_text(" ".join(s.text for s in segments[index : index + 3]))  # spans segments
        start_score = fuzz.partial_ratio(target, clean_text(segments[index].text))  # find start
        score = (fuzz.partial_ratio(target, nearby_text), start_score)
        if score > best_score:
            best_score, best_index = score, index
    if best_index is None:
        return False, None
    found = best_score[0] >= min_score
    return found, format_time(segments[best_index].start) if found else None


EMPTY_VALUES = {
    "", "none", "null", "n a", "na", "tbd", "tba", "unknown", "unspecified", "not specified",
    "not stated", "not mentioned", "unassigned", "someone", "anyone", "everyone", "all",
}
COMMON_WORDS = {"the", "a", "an", "of", "by", "on", "at", "in", "to", "and", "team", "end", "next", "this", "before"}


def is_empty_value(value: str | None) -> bool:
    return value is None or clean_text(value) in EMPTY_VALUES


def appears_in_transcript(value: str, transcript_text: str) -> bool:
    transcript = f" {clean_text(transcript_text)} "
    key_words = [w for w in clean_text(value).split() if w not in COMMON_WORDS]
    if not key_words:
        key_words = clean_text(value).split()
    return bool(key_words) and all(f" {w} " in transcript for w in key_words)
