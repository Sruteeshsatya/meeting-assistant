"""Gemini client."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import LLMError

log = logging.getLogger(__name__)
Schema = TypeVar("Schema", bound=BaseModel)

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMClient(Protocol):
    def generate(self, *, model: str, system: str, prompt: str, schema: type[Schema]) -> Schema: ...


def parse_json_response(text: str, schema: type[Schema]) -> Schema:
    cleaned = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.S)  # strip ```json
    if fenced:
        cleaned = fenced.group(1)
    return schema.model_validate(json.loads(cleaned))


class GeminiClient:
    def __init__(self, api_key: str, *, max_retries: int = 5):
        if not api_key:
            raise LLMError("No Gemini API key configured. Set GEMINI_API_KEY (see README).")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.max_retries = max_retries

    def generate(self, *, model: str, system: str, prompt: str, schema: type[Schema]) -> Schema:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,  # stay faithful
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=32768,
        )
        message = prompt
        last_problem = ""
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(model=model, contents=message, config=config)
            except Exception as exc:
                status = getattr(exc, "code", None)
                should_retry = status in RETRY_STATUS_CODES or (status is None and attempt < 2)
                if should_retry and attempt < self.max_retries - 1:
                    wait = min(2 ** (attempt + 1), 30)
                    log.warning("Gemini %s (attempt %d); retrying in %ss", status, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(friendly_error(exc, model), detail=str(exc)) from exc

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            text = getattr(response, "text", None)
            if not text:
                last_problem = f"empty response (finish reason: {get_finish_reason(response)})"
            else:
                try:
                    return parse_json_response(text, schema)
                except (ValueError, ValidationError) as exc:
                    last_problem = f"invalid JSON: {exc}"
            log.warning("Gemini returned unusable output (%s); retrying", last_problem)
            message = prompt + "\n\nYour previous reply was not valid JSON for the schema. Reply with JSON only."
        raise LLMError(f"The language model ({model}) did not return a usable result: {last_problem}.")


def get_finish_reason(response) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:
        return "unknown"


def friendly_error(exc: Exception, model: str) -> str:
    status = getattr(exc, "code", None)
    text = str(exc)
    if status in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
        return "The Gemini API key was rejected. Check GEMINI_API_KEY."
    if status == 404:
        return f"Gemini model '{model}' was not found. Choose another model in the sidebar or .env."
    if status == 429:
        return "Gemini rate limit or quota exceeded. Wait a minute and try again, or use another model."
    if status and 500 <= status < 600:
        return "Gemini is temporarily unavailable. Please retry in a moment."
    return f"Calling the Gemini model '{model}' failed: {type(exc).__name__}."
