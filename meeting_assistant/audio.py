"""Stage 1a: check and decode audio."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import MAX_FILE_MB, MIN_DURATION_SECONDS, SUPPORTED_EXTENSIONS
from .errors import AudioInputError

SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 3 * 60 * 60


def load_audio(data: bytes, filename: str) -> tuple[np.ndarray, float]:
    """Return (samples, seconds)."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise AudioInputError(
            f"Unsupported file type '{extension or 'no extension'}'. "
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    if not data:
        raise AudioInputError("The uploaded file is empty (0 bytes). Please upload a meeting recording.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise AudioInputError(f"The file is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB.")

    samples = decode_audio(data, extension)
    duration = len(samples) / SAMPLE_RATE

    if duration < MIN_DURATION_SECONDS:
        raise AudioInputError(
            f"The recording is only {duration:.1f} s long, which is too short to contain a meeting."
        )
    if duration > MAX_DURATION_SECONDS:
        raise AudioInputError("The recording is longer than 3 hours. Please split it into shorter parts.")
    if float(np.max(np.abs(samples))) < 1e-3:
        raise AudioInputError("The recording appears to be silent (no audible signal was found).")
    return samples, duration


def decode_audio(data: bytes, extension: str) -> np.ndarray:
    handle, path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(data)
        if shutil.which("ffmpeg"):
            samples = decode_with_ffmpeg(path)
        else:  # no ffmpeg (e.g. Windows)
            samples = decode_with_pyav(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if samples is None or len(samples) == 0:
        raise AudioInputError("The file contains no audio stream that could be decoded.")
    return samples


def unreadable_file_error(detail: str) -> AudioInputError:
    return AudioInputError(
        "The file could not be read as audio. It may be corrupted, encrypted, "
        "or not really an audio file despite its extension.",
        detail=detail,
    )


def decode_with_ffmpeg(path: str) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    try:
        process = subprocess.run(command, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise unreadable_file_error("ffmpeg timed out while decoding") from exc
    if process.returncode != 0:
        raise unreadable_file_error(process.stderr.decode("utf-8", "replace").strip()[-500:])
    return np.frombuffer(process.stdout, np.int16).astype(np.float32) / 32768.0


def decode_with_pyav(path: str) -> np.ndarray:
    try:
        import av

        with av.open(path) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            chunks = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
        return np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0, np.float32)
    except Exception as exc:
        raise unreadable_file_error(f"{type(exc).__name__}: {exc}") from exc
