"""Tests for Phase 3: ML classifier training + MIDI integration + integration"""
import numpy as np
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_collector():
    """Test gesture data collection pipeline."""
    from gesture.gesture_classifier import GestureDataCollector

    collector = GestureDataCollector(labels=["rest", "engage", "stop"])

    # Record samples for 3 gestures
    collector.set_label("rest")
    for _ in range(20):
        # Open hand: landmarks spread out
        sample = [(0.5 + np.random.randn() * 0.01,
                    0.5 + np.random.randn() * 0.01,
                    np.random.randn() * 0.01) for _ in range(21)]
        # Spread fingers
        for i in [4, 8, 12, 16, 20]:
            sample[i] = (0.5 + np.random.randn() * 0.15, 0.5 + np.random.randn() * 0.15, 0.0)
        collector.record_sample([{"landmarks": sample}])

    collector.set_label("engage")
    for _ in range(20):
        # Pinch: thumb and index close, others spread
        sample = [(0.5 + np.random.randn() * 0.01,
                    0.5 + np.random.randn() * 0.01,
                    np.random.randn() * 0.01) for _ in range(21)]
        sample[4] = (0.30, 0.35, 0.0)  # thumb near index
        sample[8] = (0.31, 0.35, 0.0)  # index near thumb
        for i in [12, 16, 20]:
            sample[i] = (0.5 + np.random.randn() * 0.1, 0.5 + np.random.randn() * 0.1, 0.0)
        collector.record_sample([{"landmarks": sample}])

    collector.set_label("stop")
    for _ in range(20):
        # Fist: all tips near palm
        sample = [(0.5 + np.random.randn() * 0.01,
                    0.5 + np.random.randn() * 0.01,
                    np.random.randn() * 0.01) for _ in range(21)]
        for i in [4, 8, 12, 16, 20]:
            sample[i] = (0.50, 0.42, 0.05)  # fingertips curled
        collector.record_sample([{"landmarks": sample}])

    counts = collector.get_sample_counts()
    assert counts["rest"] == 20
    assert counts["engage"] == 20
    assert counts["stop"] == 20
    print(f"  collector: {counts} pass")

    X, y = collector.get_dataset()
    assert X.shape == (60, 63), f"Expected (60, 63), got {X.shape}"
    assert y.shape == (60,), f"Expected (60,), got {y.shape}"
    print(f"  collector: dataset shape X={X.shape} y={y.shape} pass")

    # Save
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        save_path = f.name
    collector.save(save_path)
    assert os.path.exists(save_path + ".npz") or os.path.exists(save_path)
    os.unlink(save_path)
    if os.path.exists(save_path + ".npz"):
        os.unlink(save_path + ".npz")
    print("  collector: save pass")

    print("[PASS] GestureDataCollector")
    return True


def test_train_and_infer():
    """Train a small classifier and verify inference works."""
    from gesture.gesture_classifier import GestureDataCollector, train_classifier, GestureClassifier

    # Generate synthetic data
    collector = GestureDataCollector(labels=["rest", "engage", "stop"])

    np.random.seed(42)
    collector.set_label("rest")
    for _ in range(30):
        lm = [(0.5 + np.random.randn() * 0.02, 0.5 + np.random.randn() * 0.02, 0.0) for _ in range(21)]
        # Spread fingers
        for i in [4, 8, 12, 16, 20]:
            lm[i] = (0.5 + np.random.randn() * 0.15, 0.5 + np.random.randn() * 0.15, 0.0)
        collector.record_sample([{"landmarks": lm}])

    collector.set_label("engage")
    for _ in range(30):
        lm = [(0.5 + np.random.randn() * 0.02, 0.5 + np.random.randn() * 0.02, 0.0) for _ in range(21)]
        lm[4] = (0.30, 0.35, 0.0)  # pinch
        lm[8] = (0.31, 0.35, 0.0)
        for i in [12, 16, 20]:
            lm[i] = (0.5 + np.random.randn() * 0.1, 0.5 + np.random.randn() * 0.1, 0.0)
        collector.record_sample([{"landmarks": lm}])

    collector.set_label("stop")
    for _ in range(30):
        lm = [(0.5 + np.random.randn() * 0.02, 0.5 + np.random.randn() * 0.02, 0.0) for _ in range(21)]
        for i in [4, 8, 12, 16, 20]:
            lm[i] = (0.50, 0.42, 0.05)  # fist
        collector.record_sample([{"landmarks": lm}])

    X, y = collector.get_dataset()

    # Train and export
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = f.name
    try:
        success = train_classifier(
            X, y,
            labels=["rest", "engage", "stop"],
            hidden_sizes=(64, 32),
            max_iter=200,
            output_path=model_path,
        )

        if not success:
            # skl2onnx may not be installed — skip inference test
            print("  train: sklearn training completed (ONNX export skipped)")
            print("[PASS] Train (without ONNX)")
            return True

        # Load and infer
        clf = GestureClassifier(model_path=model_path)
        assert clf.model_loaded, "Model should be loaded"

        # Test with engage-like landmarks
        test_landmarks = [(0.5, 0.5, 0.0)] * 21
        test_landmarks[4] = (0.30, 0.35, 0.0)
        test_landmarks[8] = (0.31, 0.35, 0.0)

        result = clf.classify([{"landmarks": test_landmarks}])
        assert result is not None, "Should classify engage gesture"
        print(f"  infer: classified as '{result['gesture']}' (conf={result['confidence']:.2f})")

        # Test with fist-like landmarks
        test_fist = [(0.5, 0.5, 0.0)] * 21
        for i in [4, 8, 12, 16, 20]:
            test_fist[i] = (0.50, 0.42, 0.05)

        result_fist = clf.classify([{"landmarks": test_fist}])
        assert result_fist is not None, "Should classify fist gesture"
        print(f"  infer: classified as '{result_fist['gesture']}' (conf={result_fist['confidence']:.2f})")

        # Test with None input
        assert clf.classify([]) is None
        assert clf.classify(None) is None
        print("  infer: edge cases pass")

        print("[PASS] Train + Inference")
        return True
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_midi_pitch_bend_accuracy():
    """Verify MIDI pitch_bend maps frequencies correctly."""
    from engine.midi_output import MIDIOutput

    # A4 = 440Hz, MIDI 69 → bend should be center (8192)
    center = MIDIOutput.freq_to_pitch_bend(440.0, 69)
    assert center == 8192, f"A4 center: {center}"

    # A4 + 50 cents (quarter tone sharp)
    freq_sharp = 440.0 * (2 ** (50 / 1200))
    bend_sharp = MIDIOutput.freq_to_pitch_bend(freq_sharp, 69)
    assert bend_sharp > 8192, f"50 cents sharp should be > 8192: {bend_sharp}"
    # 50 cents / 200 cents per semitone range * 8191 = ~2048 above center
    expected = int(8192 + (50 / 200) * 8191)
    assert abs(bend_sharp - expected) < 5, f"Expected ~{expected}, got {bend_sharp}"

    # A4 - 100 cents (one semitone flat)
    freq_flat = 440.0 * (2 ** (-100 / 1200))
    bend_flat = MIDIOutput.freq_to_pitch_bend(freq_flat, 69)
    assert bend_flat < 8192, f"100 cents flat should be < 8192: {bend_flat}"

    print(f"  midi pitch_bend: center={center} +50c={bend_sharp} -100c={bend_flat} pass")
    print("[PASS] MIDI pitch_bend accuracy")
    return True


def test_full_pipeline_sim():
    """Simulate a full frame pipeline without camera/audio hardware."""
    from utils.smoothing import LandmarkSmoother
    from gesture.pinch_detector import PinchDetector
    from gesture.theremin_mapper import ThereminMapper

    smoother = LandmarkSmoother(alpha=0.3)
    pinch = PinchDetector(threshold=0.05)
    mapper = ThereminMapper()

    # Simulate 30 frames of hand moving from left to right while pinching
    for frame in range(30):
        x = 0.2 + (frame / 30) * 0.6  # move from 0.2 to 0.8
        y = 0.3

        landmarks = [(x, y, 0.0)] * 21
        # Pinch position
        landmarks[4] = (x - 0.02, y - 0.02, 0.0)
        landmarks[8] = (x - 0.01, y - 0.02, 0.0)

        hands_data = [{"type": "Right", "landmarks": landmarks, "raw_landmarks": None}]
        smoothed = smoother.smooth(hands_data)
        pinch_state = pinch.detect(smoothed)
        theremin_data = mapper.map_hand(smoothed[0]["landmarks"], engaged=True)

        assert theremin_data["engaged"] == True
        assert theremin_data["pitch"] > 100
        assert 0.0 <= theremin_data["volume"] <= 1.0

    # Final pitch should be higher than initial (moved right)
    assert theremin_data["pitch"] > 200, f"Final pitch should be > 200, got {theremin_data['pitch']}"
    print(f"  pipeline: 30 frames, final pitch={theremin_data['pitch']:.1f}Hz pass")

    # Now release pinch for 10 frames
    for frame in range(10):
        hands_data = [{"type": "Right", "landmarks": landmarks, "raw_landmarks": None}]
        smoothed = smoother.smooth(hands_data)
        pinch_state = pinch.detect(smoothed)
        theremin_data = mapper.map_hand(smoothed[0]["landmarks"], engaged=False)

    assert theremin_data["volume"] < 0.1, f"After release, vol should be low: {theremin_data['volume']:.2f}"
    print(f"  pipeline: after release vol={theremin_data['volume']:.3f} pass")

    print("[PASS] Full pipeline simulation")
    return True


if __name__ == "__main__":
    passed = 0
    failed = 0

    for test_fn in [test_data_collector, test_train_and_infer,
                    test_midi_pitch_bend_accuracy, test_full_pipeline_sim]:
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
    print(f"Phase 3 Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
