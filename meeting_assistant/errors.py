"""Errors shown in the UI."""


class PipelineError(Exception):
    stage = "pipeline"

    def __init__(self, user_message: str, *, detail: str | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class AudioInputError(PipelineError):
    stage = "upload"


class TranscriptionError(PipelineError):
    stage = "transcription"


class LLMError(PipelineError):
    stage = "language model"


class RefinementError(LLMError):
    stage = "transcript refinement"


class DocumentationError(LLMError):
    stage = "meeting documentation"
