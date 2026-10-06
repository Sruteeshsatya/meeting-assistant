"""App settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RUNS_DIR = PROJECT_ROOT / "runs"

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac", ".webm", ".mp4",
}
MAX_FILE_MB = 500
MIN_DURATION_SECONDS = 1.0


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    refine_model: str = field(default_factory=lambda: env("REFINE_MODEL", "gemini-3.5-flash-lite"))
    docs_model: str = field(default_factory=lambda: env("DOCS_MODEL", "gemini-3.5-flash-lite"))
    whisper_model: str = field(default_factory=lambda: env("WHISPER_MODEL", "auto"))
    use_glossary_hint: bool = field(default_factory=lambda: env("STT_USE_GLOSSARY", "1") != "0")  # terms -> Whisper
    words_per_chunk: int = 3000
