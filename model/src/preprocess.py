"""
Audio preprocessing for the SSL (Wav2Vec2) model.

Unlike the earlier CNN version, Wav2Vec2 consumes raw waveform directly
(not spectrograms) — it has its own internal feature-learning front end.
Kept as plain functions so training, evaluation, and live streaming all
use the exact same pipeline.
"""

import numpy as np
import librosa

SAMPLE_RATE = 16000
FIXED_DURATION_SEC = 4
FIXED_LEN_SAMPLES = SAMPLE_RATE * FIXED_DURATION_SEC


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def pad_or_crop(audio: np.ndarray, length: int = FIXED_LEN_SAMPLES) -> np.ndarray:
    if len(audio) >= length:
        start = max(0, (len(audio) - length) // 2)
        return audio[start:start + length]
    pad_width = length - len(audio)
    return np.pad(audio, (0, pad_width), mode="wrap")


def normalize(audio: np.ndarray) -> np.ndarray:
    """Wav2Vec2 expects roughly zero-mean, unit-variance input."""
    return (audio - audio.mean()) / (audio.std() + 1e-6)


def audio_file_to_waveform(path: str) -> np.ndarray:
    """File path -> ready-to-model waveform tensor, shape (FIXED_LEN_SAMPLES,)."""
    audio = load_audio(path)
    audio = pad_or_crop(audio)
    return normalize(audio).astype(np.float32)


def waveform_to_model_input(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Same pipeline but for in-memory audio (used by the live streaming path)."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = pad_or_crop(audio)
    return normalize(audio).astype(np.float32)
