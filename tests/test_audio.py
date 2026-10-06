import pytest

from meeting_assistant.audio import load_audio
from meeting_assistant.errors import AudioInputError
from tests.fakes import wav_bytes


def test_valid_wav_decodes():
    samples, duration = load_audio(wav_bytes(2.0), "meeting.wav")
    assert abs(duration - 2.0) < 0.05
    assert samples.dtype.name == "float32"


def test_decodes_without_ffmpeg_binary(monkeypatch):
    # e.g. Windows without ffmpeg
    import meeting_assistant.audio as audio

    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    _, duration = load_audio(wav_bytes(2.0), "meeting.wav")
    assert abs(duration - 2.0) < 0.05
    with pytest.raises(AudioInputError, match="could not be read"):
        load_audio(b"junk" * 500, "meeting.mp3")


@pytest.mark.parametrize(
    "data,name,expected",
    [
        (wav_bytes(), "notes.txt", "Unsupported file type"),
        (wav_bytes(), "noextension", "Unsupported file type"),
        (b"", "meeting.mp3", "empty"),
        (b"this is definitely not audio" * 50, "meeting.mp3", "could not be read"),
        (wav_bytes(0.3), "short.wav", "too short"),
        (wav_bytes(2.0, amplitude=0.0), "silent.wav", "silent"),
    ],
    ids=["unsupported-ext", "no-ext", "empty", "garbage", "too-short", "silent"],
)
def test_bad_inputs_have_clear_errors(data, name, expected):
    with pytest.raises(AudioInputError) as exc:
        load_audio(data, name)
    assert expected in exc.value.user_message
