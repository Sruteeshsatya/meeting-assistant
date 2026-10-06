"""Stage 1b: speech-to-text (faster-whisper)."""

from __future__ import annotations

import ctypes
import glob
import logging
import os
from typing import Callable, Optional, Protocol

import numpy as np

from .errors import TranscriptionError
from .schema import Segment, Transcript

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

CPU_FALLBACK_MODEL = "small.en"  # fast enough on CPU


class SpeechToText(Protocol):
    name: str

    def transcribe(
        self, samples: np.ndarray, duration: float, *, hint: str = "", progress: Optional[ProgressCallback] = None
    ) -> Transcript: ...


def load_gpu_libraries() -> None:
    # load Colab CUDA libraries
    try:
        import nvidia
    except ImportError:
        return
    library_files: list[str] = []
    for base in getattr(nvidia, "__path__", []):
        for name in ("cuda_runtime", "cublas", "cudnn"):
            library_files += sorted(glob.glob(os.path.join(base, name, "lib", "lib*.so*")))
    pending = library_files
    for _ in range(3):  # retry for dependencies
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if not failed or len(failed) == len(pending):
            break
        pending = failed


def pick_device() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # pragma: no cover
        pass
    return "cpu", "int8"


def choose_model_name(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    return "large-v3" if device == "cuda" else "small.en"


class WhisperTranscriber:
    """Load once, reuse."""

    def __init__(self, model_name: str = "auto"):
        device, compute_type = pick_device()
        if device == "cuda":
            load_gpu_libraries()
        self.model_name = choose_model_name(model_name, device)
        try:
            self._model = self.load_model(self.model_name, device, compute_type)
        except Exception as exc:
            if device != "cuda":
                raise self.model_load_error(exc) from exc
            log.warning("GPU load failed (%s); using CPU", exc)
            device, compute_type = "cpu", "int8"
            if model_name in ("", "auto"):
                self.model_name = CPU_FALLBACK_MODEL
            try:
                self._model = self.load_model(self.model_name, device, compute_type)
            except Exception as cpu_exc:
                raise self.model_load_error(cpu_exc) from cpu_exc
        self.device = device
        self.name = f"faster-whisper {self.model_name} ({device})"

    @staticmethod
    def load_model(name: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        return WhisperModel(name, device=device, compute_type=compute_type)

    def model_load_error(self, exc: Exception) -> TranscriptionError:
        return TranscriptionError(
            f"Could not load the speech-to-text model '{self.model_name}'. On first use it is downloaded "
            "from Hugging Face, so check the internet connection and the model name.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    def run_whisper(self, samples, duration, hint, progress) -> tuple[list[Segment], str]:
        whisper_segments, info = self._model.transcribe(
            samples,
            language="en",
            beam_size=5,
            vad_filter=True,  # skip silence
            condition_on_previous_text=False,  # avoid repetition loops
            initial_prompt=hint[:800] or None,  # domain terms
        )
        segments: list[Segment] = []
        for piece in whisper_segments:
            text = piece.text.strip()
            if text:
                segments.append(Segment(id=len(segments), start=piece.start, end=piece.end, text=text))
            if progress and duration:
                progress(min(piece.end / duration, 1.0), f"Transcribed {piece.end:.0f}s of {duration:.0f}s")
        return segments, info.language

    def transcribe(
        self, samples: np.ndarray, duration: float, *, hint: str = "", progress: Optional[ProgressCallback] = None
    ) -> Transcript:
        try:
            segments, language = self.run_whisper(samples, duration, hint, progress)
        except Exception as exc:
            if self.device != "cuda":
                raise TranscriptionError(
                    "Speech-to-text failed while processing the audio.", detail=f"{type(exc).__name__}: {exc}"
                ) from exc
            # GPU failed, retry on CPU
            gpu_error = f"{type(exc).__name__}: {exc}"
            log.warning("GPU transcription failed (%s); retrying on CPU", gpu_error)
            if progress:
                progress(0.0, f"GPU failed - retrying on CPU with {CPU_FALLBACK_MODEL} (slower)")
            try:
                self._model = self.load_model(CPU_FALLBACK_MODEL, "cpu", "int8")
                self.model_name, self.device = CPU_FALLBACK_MODEL, "cpu"
                self.name = f"faster-whisper {CPU_FALLBACK_MODEL} (cpu fallback - GPU failed)"
                segments, language = self.run_whisper(samples, duration, hint, progress)
            except Exception as cpu_exc:
                raise TranscriptionError(
                    "Speech-to-text failed on both the GPU and the CPU.",
                    detail=f"GPU: {gpu_error}\nCPU: {type(cpu_exc).__name__}: {cpu_exc}",
                ) from cpu_exc

        if not segments:
            raise TranscriptionError(
                "No speech was detected in this recording. Check that it contains spoken English."
            )
        return Transcript(segments=segments, duration=duration, language=language, model=self.name)
