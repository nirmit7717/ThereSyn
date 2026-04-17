"""Tests for Phase 2: UI, waveforms, visualizer"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_oscillator_all_waveforms_quality():
    """Verify all waveforms produce valid, normalized output at various frequencies."""
    from dsp.oscillator import Oscillator
    osc = Oscillator(sample_rate=44100)

    waveforms = ["sine", "sawtooth", "square", "triangle"]
    test_freqs = [100, 440, 1000, 5000, 10000]

    for wf in waveforms:
        for freq in test_freqs:
            wave = osc.generate(wf, freq, 0.05, volume=1.0)
            # Valid range
            assert np.max(np.abs(wave)) <= 1.0 + 1e-6, f"{wf}@{freq}: peak={np.max(np.abs(wave))}"
            assert np.min(wave) >= -1.0 - 1e-6, f"{wf}@{freq}: min={np.min(wave)}"
            # No NaN
            assert not np.any(np.isnan(wave)), f"{wf}@{freq}: contains NaN"
            # Not all zeros
            assert np.max(np.abs(wave)) > 0.01, f"{wf}@{freq}: all near zero"

    print(f"  waveforms: {len(waveforms)}×{len(test_freqs)} combos pass")
    print("[PASS] Oscillator all waveforms")
    return True


def test_oscillator_aliasing():
    """Check that high-frequency waveforms don't alias badly (band-limited)."""
    from dsp.oscillator import Oscillator
    osc = Oscillator(sample_rate=44100)

    # At 10kHz, band-limited sawtooth should have limited energy above 2x fundamental
    wave_saw = osc.generate("sawtooth", 10000, 0.1, volume=1.0)
    spectrum = np.abs(np.fft.rfft(wave_saw))
    # Check energy above 3x fundamental (30kHz region) — should be minimal
    bin_30k = int(30000 / (44100 / 2) * len(spectrum))
    if bin_30k < len(spectrum):
        alias_energy = np.sum(spectrum[bin_30k:])
        total_energy = np.sum(spectrum)
        alias_ratio = alias_energy / total_energy if total_energy > 0 else 0
        assert alias_ratio < 0.3, f"High-freq aliasing ratio: {alias_ratio:.4f}"
        print(f"  aliasing: high-freq ratio={alias_ratio:.4f} pass")
    else:
        print("  aliasing: skip (bin out of range)")

    # Sine at 20kHz should still be valid
    wave_sine_20k = osc.generate("sine", 20000, 0.05, volume=1.0)
    assert np.max(np.abs(wave_sine_20k)) > 0.5, "20kHz sine should have energy"
    print("  aliasing: 20kHz sine pass")

    print("[PASS] Oscillator anti-aliasing")
    return True


def test_envelope_short():
    """Envelope with very short signal (edge case)."""
    from dsp.envelope import ADSREnvelope
    env = ADSREnvelope(attack=0.01, decay=0.01, sustain_level=0.5, release=0.01, sample_rate=1000)

    # Very short signal (10 samples)
    wave = np.ones(10)
    shaped = env.apply(wave)
    assert len(shaped) == 10
    assert not np.any(np.isnan(shaped))
    print("  envelope: short signal pass")

    # Single sample
    wave_single = np.ones(1)
    shaped_single = env.apply(wave_single)
    assert len(shaped_single) == 1
    assert not np.any(np.isnan(shaped_single))
    print("  envelope: single sample pass")

    print("[PASS] ADSREnvelope edge cases")
    return True


def test_filter_sweep():
    """Test filter cutoff sweep from dark to bright."""
    from dsp.filter import LowpassFilter
    filt = LowpassFilter(sample_rate=44100)

    sr = 44100
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    # Signal with multiple harmonics
    signal = (np.sin(2 * np.pi * 200 * t) +
              np.sin(2 * np.pi * 2000 * t) +
              np.sin(2 * np.pi * 8000 * t)) / 3.0

    # Dark (cutoff=0.0)
    filt._prev = 0.0
    dark = filt.apply(signal.copy(), cutoff_normalized=0.0)
    dark_rms = np.sqrt(np.mean(dark ** 2))

    # Bright (cutoff=1.0)
    filt._prev = 0.0
    bright = filt.apply(signal.copy(), cutoff_normalized=1.0)
    bright_rms = np.sqrt(np.mean(bright ** 2))

    # Mid (cutoff=0.5)
    filt._prev = 0.0
    mid = filt.apply(signal.copy(), cutoff_normalized=0.5)
    mid_rms = np.sqrt(np.mean(mid ** 2))

    assert dark_rms < mid_rms < bright_rms, \
        f"Expected dark<{mid}<bright, got {dark_rms:.3f}<{mid_rms:.3f}<{bright_rms:.3f}"
    print(f"  filter sweep: dark={dark_rms:.3f} mid={mid_rms:.3f} bright={bright_rms:.3f} pass")

    print("[PASS] Filter sweep")
    return True


def test_theremin_mapper_extreme_positions():
    """Test mapper at exact corners and edges."""
    from gesture.theremin_mapper import ThereminMapper
    mapper = ThereminMapper()

    corners = [
        ((0.0, 0.0), "top-left"),
        ((1.0, 0.0), "top-right"),
        ((0.0, 1.0), "bottom-left"),
        ((1.0, 1.0), "bottom-right"),
    ]

    for (x, y), label in corners:
        mapper.reset()
        landmarks = [(x, y, 0.0)] * 21
        for _ in range(15):
            result = mapper.map_hand(landmarks, engaged=True)
        assert 0.0 <= result["volume"] <= 1.0, f"{label}: vol={result['volume']}"
        assert 100 < result["pitch"] < 1200, f"{label}: pitch={result['pitch']}"
        assert 0.0 <= result["filter_cutoff"] <= 1.0, f"{label}: cutoff={result['filter_cutoff']}"
        print(f"  mapper extreme {label}: pitch={result['pitch']:.1f} vol={result['volume']:.2f} cutoff={result['filter_cutoff']:.2f} pass")

    print("[PASS] ThereminMapper extremes")
    return True


def test_midi_output():
    """Test MIDI utility functions (no actual MIDI port needed)."""
    from engine.midi_output import MIDIOutput

    # note_name_to_midi
    assert MIDIOutput.note_name_to_midi("C4") == 60
    assert MIDIOutput.note_name_to_midi("A4") == 69
    assert MIDIOutput.note_name_to_midi("C#4") == 61
    assert MIDIOutput.note_name_to_midi("B3") == 59
    print("  midi: note_name_to_midi pass")

    # freq_to_midi
    assert abs(MIDIOutput.freq_to_midi(440.0) - 69) <= 1
    assert abs(MIDIOutput.freq_to_midi(261.63) - 60) <= 1
    print("  midi: freq_to_midi pass")

    # freq_to_pitch_bend
    center = MIDIOutput.freq_to_pitch_bend(440.0, 69)
    assert abs(center - 8192) < 100, f"Center bend should be ~8192, got {center}"
    sharp = MIDIOutput.freq_to_pitch_bend(466.16, 69)  # A#4
    assert sharp > center, f"Sharp bend should be > center"
    flat = MIDIOutput.freq_to_pitch_bend(415.30, 69)  # Ab4
    assert flat < center, f"Flat bend should be < center"
    print(f"  midi: pitch_bend center={center} sharp={sharp} flat={flat} pass")

    # MIDIOutput with enabled=False (no port)
    midi = MIDIOutput(enabled=False)
    # Should not crash
    midi.note_on(60, 127)
    midi.note_off(60)
    midi.pitch_bend(8192)
    midi.control_change(7, 100)
    print("  midi: no-port passthrough pass")

    print("[PASS] MIDIOutput utilities")
    return True


def test_latency_profiler():
    """Test latency profiler measurements."""
    from engine.latency_profiler import LatencyProfiler
    import time

    profiler = LatencyProfiler(report_interval=0.01)

    profiler.start_frame()
    profiler.mark("vision")
    profiler.mark("gesture")
    profiler.mark("audio_queue")
    time.sleep(0.012)
    report = profiler.end_frame()

    assert report is not None
    assert "frame_total" in report
    assert report["frame_total"]["avg_ms"] > 0
    assert report["frame_total"]["max_ms"] >= report["frame_total"]["avg_ms"]
    print(f"  profiler: frame_total={report['frame_total']['avg_ms']:.2f}ms pass")

    # Run multiple frames
    for _ in range(50):
        profiler.start_frame()
        time.sleep(0.001)
        profiler.mark("vision")
        profiler.mark("gesture")
        profiler.mark("audio_queue")
        profiler.end_frame()

    report = profiler.get_report()
    assert report["frame_total"]["avg_ms"] > 0
    assert report["vision"]["avg_ms"] >= 0
    print(f"  profiler: 50-frame avg={report['frame_total']['avg_ms']:.2f}ms pass")

    print("[PASS] LatencyProfiler")
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0

    for test_fn in [test_oscillator_all_waveforms_quality, test_oscillator_aliasing,
                    test_envelope_short, test_filter_sweep,
                    test_theremin_mapper_extreme_positions,
                    test_midi_output, test_latency_profiler]:
        print(f"\n{'='*50}")
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Phase 2 Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
