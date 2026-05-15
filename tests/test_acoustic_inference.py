from __future__ import annotations

import io
import wave

import numpy as np

from keyboard_fusion.acoustic_inference import fit_spectrogram_shape, read_wav_bytes_mono_float, resample_linear


def wav_bytes(samples: np.ndarray, sample_rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        pcm = np.clip(samples, -1.0, 1.0)
        wav_file.writeframes((pcm * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_read_wav_bytes_mono_float() -> None:
    raw = wav_bytes(np.array([0.0, 0.5, -0.5], dtype=np.float32), sample_rate=8000)

    sample_rate, samples = read_wav_bytes_mono_float(raw)

    assert sample_rate == 8000
    assert samples.shape == (3,)
    assert np.isclose(samples[1], 0.5, atol=1e-3)


def test_resample_linear_changes_sample_count() -> None:
    samples = np.linspace(0.0, 1.0, 10, dtype=np.float32)

    resampled = resample_linear(samples, source_rate=10, target_rate=20)

    assert len(resampled) == 20
    assert np.isclose(resampled[0], 0.0)
    assert np.isclose(resampled[-1], 1.0, atol=0.06)


def test_fit_spectrogram_shape_pads_and_crops() -> None:
    spectrogram = np.ones((2, 5), dtype=np.float32)

    fitted = fit_spectrogram_shape(spectrogram, (4, 3))

    assert fitted.shape == (4, 3)
    assert np.all(fitted[:2, :3] == 1.0)
    assert np.all(fitted[2:, :] == 0.0)
