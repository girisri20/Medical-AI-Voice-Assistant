"""Audio utility helpers for the Medical Speech-to-Text demo."""

from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """Validate that an audio file exists, is readable, and has a supported format.

    Returns:
        (is_valid, message)
    """
    if not file_path:
        return False, "No audio file path provided."

    if not os.path.exists(file_path):
        return False, f"Audio file not found: {file_path}"

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        return False, (
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    try:
        info = sf.info(file_path)
    except Exception as exc:  # pragma: no cover - depends on file contents
        return False, f"Unable to read audio file: {exc}"

    if info.frames == 0:
        return False, "Audio file is empty (no audio frames)."

    return True, "Audio file is valid."


def read_audio(file_path: str) -> Optional[Tuple["any", int]]:
    """Read an audio file and return (samples, sample_rate).

    Samples are returned as a numpy array via soundfile.
    Returns None if reading fails.
    """
    try:
        samples, sample_rate = sf.read(file_path, dtype="float32")
        return samples, sample_rate
    except Exception as exc:  # pragma: no cover - depends on file contents
        print(f"[audio_utils] Failed to read audio file: {exc}")
        return None


def bytes_to_wav_file(audio_bytes: bytes, output_path: str) -> bool:
    """Write raw audio bytes to a WAV file.

    Returns True on success, False on failure.
    """
    try:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[audio_utils] Failed to write WAV file: {exc}")
        return False


def audio_bytes_info(audio_bytes: bytes) -> Tuple[bool, str]:
    """Basic validation of in-memory audio bytes.

    Returns:
        (is_valid, message)
    """
    if not audio_bytes:
        return False, "No audio data provided."
    if len(audio_bytes) < 100:
        return False, "Audio data is too small to contain valid audio."
    return True, f"Audio bytes received ({len(audio_bytes)} bytes)."


def get_audio_duration(file_path: str) -> Optional[float]:
    """Return the duration of an audio file in seconds, or None on failure."""
    try:
        info = sf.info(file_path)
        if info.frames == 0 or info.samplerate == 0:
            return None
        return info.frames / info.samplerate
    except Exception:
        return None
