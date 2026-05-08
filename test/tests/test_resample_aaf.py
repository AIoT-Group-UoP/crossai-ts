"""Regression tests for issue #105: AAF not Considered when Resampling.

Verifies that ``resample_2d`` applies an anti-aliasing filter when downsampling.
A pure tone above the *target* Nyquist frequency must be attenuated, not folded
back into the audible band.
"""
import numpy as np
import pytest

from caits.preprocessing import resample_2d, resample_signal


def _tone(freq_hz: float, sr: int, duration_s: float = 1.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t).astype("float64")


def _energy_at(sig: np.ndarray, freq_hz: float, sr: int, bw_hz: float = 50.0) -> float:
    """RMS amplitude in a narrow band around freq_hz."""
    n = len(sig)
    spectrum = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= freq_hz - bw_hz) & (freqs <= freq_hz + bw_hz)
    if not mask.any():
        return 0.0
    # Parseval-style band energy, normalized to a comparable scale.
    return float(np.sqrt(np.sum(np.abs(spectrum[mask]) ** 2)) / n)


def test_resample_2d_suppresses_aliasing_on_downsample():
    """A 7 kHz tone in a 16 kHz signal downsampled to 8 kHz must NOT alias to 1 kHz.

    Target Nyquist = 4 kHz; 7 kHz would fold to |7 - 8| = 1 kHz without an AAF.
    """
    native_sr = 16_000
    target_sr = 8_000
    tone_freq = 7_000  # above target Nyquist (4 kHz)
    alias_freq = abs(tone_freq - target_sr)  # 1 kHz

    sig = _tone(tone_freq, native_sr, duration_s=1.0)

    out = resample_2d(sig, native_sr=native_sr, target_sr=target_sr, dtype="float64")
    assert out.shape[1] == 1
    out_1d = out[:, 0]

    alias_energy = _energy_at(out_1d, alias_freq, target_sr)
    # Reference: same tone at the alias frequency, same length, so we can compare scales.
    ref = _tone(alias_freq, target_sr, duration_s=len(out_1d) / target_sr)
    ref_energy = _energy_at(ref, alias_freq, target_sr)

    # AAF should knock the alias down by at least 40 dB vs. a real tone of the same amplitude.
    ratio_db = 20 * np.log10(max(alias_energy, 1e-12) / max(ref_energy, 1e-12))
    assert ratio_db < -40, (
        f"Alias at {alias_freq} Hz not sufficiently suppressed: "
        f"{ratio_db:.1f} dB vs. reference tone (expected < -40 dB)"
    )


def test_resample_signal_legacy_aliases_as_documented():
    """Sanity check: the legacy np.interp path DOES alias.

    Locks in that the new ``resample_2d`` path is doing real work — if this test
    starts failing (legacy no longer aliases), something has changed in the
    legacy implementation and the contrast assertion above may be invalidated.
    """
    native_sr = 16_000
    target_sr = 8_000
    tone_freq = 7_000
    alias_freq = abs(tone_freq - target_sr)

    sig = _tone(tone_freq, native_sr, duration_s=1.0)
    out = resample_signal(sig, native_sr=native_sr, target_sr=target_sr, dtype="float64")

    alias_energy = _energy_at(out, alias_freq, target_sr)
    # The legacy path produces measurable energy at the alias bin.
    assert alias_energy > 1e-3, (
        f"Legacy resample_signal unexpectedly suppressed the alias bin "
        f"({alias_energy:.2e}); test premise may be stale."
    )


def test_resample_2d_preserves_inband_tone():
    """A tone well below the target Nyquist must survive resampling intact."""
    native_sr = 16_000
    target_sr = 8_000
    tone_freq = 1_000  # well below target Nyquist (4 kHz)

    sig = _tone(tone_freq, native_sr, duration_s=1.0)
    out = resample_2d(sig, native_sr=native_sr, target_sr=target_sr, dtype="float64")
    out_1d = out[:, 0]

    inband = _energy_at(out_1d, tone_freq, target_sr)
    assert inband > 1e-2, f"In-band tone lost after resampling: energy={inband:.2e}"


@pytest.mark.parametrize("native_sr,target_sr", [(48_000, 16_000), (44_100, 22_050)])
def test_resample_2d_shape_and_dtype(native_sr, target_sr):
    """Output shape matches the resampling ratio and dtype is honored."""
    duration = 0.5
    sig = np.random.default_rng(0).standard_normal(int(native_sr * duration))
    out = resample_2d(sig, native_sr=native_sr, target_sr=target_sr, dtype="float32")
    expected_n = int(np.ceil(len(sig) * target_sr / native_sr))
    assert out.shape == (expected_n, 1)
    assert out.dtype == np.float32
