# ThereSyn — Build Plan (2 People, ~2 Days)

## Overall Progress

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Core Pipeline | ✅ Complete | 23/23 pass |
| Phase 2: UI + Filters | ✅ Complete | 23/23 pass |
| Phase 3: ML + MIDI + Latency | ✅ Complete | 23/23 pass |
| Phase 4: Integration + Demo | ⚠️ In Progress | See below |

---

## Phase 1: Core Pipeline — ✅ COMPLETE

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Camera + HandTracker | Nirmit | ✅ Done | Robustness additions added |
| EMA Smoothing | Collaborator | ✅ Done | Multi-hand, hand loss tested |
| Pinch Detector | Collaborator | ✅ Done | Edge detection (just_pinched, just_released) |
| Theremin Mapper | Both | ✅ Done | Deadzones, exponential pitch, filter cutoff |
| DSP Oscillator | Collaborator | ✅ Done | 4 waveforms, band-limited, anti-aliased |
| ADSR Envelope | Collaborator | ✅ Done | Edge cases covered |
| Audio Engine | Nirmit | ✅ Done | Threaded, DI for headless, error handling |
| Main Loop Wiring | Both | ✅ Done | ML octave control, null-safety on MIDI |

---

## Phase 2: UI + Polish + Filters — ✅ COMPLETE

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Theremin UI (freq, vol, pinch indicator) | Collaborator | ✅ Done | Panel, bars, engagement ring |
| Landmark overlay on frame | Nirmit | ✅ Done | Working in main.py |
| Lowpass Filter | Collaborator | ✅ Done | Sweep tested |
| Additional waveforms (saw, square, tri) | Collaborator | ✅ Done | Band-limited anti-aliased |
| Waveform cycling (W key + indicator) | Both | ✅ Done | Working in main.py |
| Audio Visualizer | Collaborator | ✅ Done | Synthetic waveform + spectrum from theremin params, fed in main loop |

---

## Phase 3: ML + MIDI + Latency — ✅ COMPLETE

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| MIDI output (pitch_bend + CC) | Both | ✅ Done | Configurable bend range |
| MIDI wiring in main loop | Nirmit | ✅ Done | Null-safety, note_on/note_off |
| Latency Profiler | Nirmit | ✅ Done | Avg/max/p95 per stage |
| ML Data Collector | Collaborator | ✅ Done | Record, save, load samples |
| ML Gesture Classifier | Both | ✅ Done | ONNX inference with metadata labels |
| ML wiring in main loop | Nirmit | ✅ Done | Auto-engage, stop, octave up/down |
| ONNX Export + Training Script | Collaborator | ✅ Done | `train_model.py` — collect, train, verify |
| Gesture labels embedded in ONNX | Collaborator | ✅ Done | Model metadata carries label list |

---

## Phase 4: Integration + Demo — ⚠️ IN PROGRESS

### ✅ Done by Collaborator (this session)

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Fix Audio Visualizer | Collaborator | ✅ Done | Synthetic waveform/spectrum, `update_from_theremin()` wired in main.py |
| Fix mediapipe import in loop | Collaborator | ✅ Done | Moved to top-level import |
| Fix camera disconnect crash | Collaborator | ✅ Done | 30-frame tolerance, graceful exit |
| Fix duplicate pygame.init() | Collaborator | ✅ Done | Removed redundant init calls |
| Add FPS counter | Collaborator | ✅ Done | Top-right corner |
| Add waveform label | Collaborator | ✅ Done | Shows current waveform name |
| Fix ONNX export | Collaborator | ✅ Done | Uses `FloatTensorType` from skl2onnx, zipmap disabled, labels in metadata |
| Fix ONNX inference | Collaborator | ✅ Done | Handles both raw prob array and zipmap output |
| Training script | Collaborator | ✅ Done | `train_model.py --collect/--train/--verify` |
| Fix floating-point test boundary | Collaborator | ✅ Done | `99.9` instead of `100.0` |

### ✅ Done by Nirmit (this session)

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Fix train_model.py drawing utils bug | Nirmit | ✅ Done | Replaced np.solutions.* with mediapipe and added import guard |
| Make sklearn-safe training path | Nirmit | ✅ Done | GestureClassifier.train skips when sklearn missing; tests pass headlessly |
| UI FPS counter hookup | Nirmit | ✅ Done | main passes clock.get_fps() to UI |
| Headless smoke-run script + test | Nirmit | ✅ Done | headless_smoke.py and tests/test_headless_smoke.py added for CI validation |

### ❌ Remaining (Nirmit's Part)

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| End-to-end testing with real camera | Nirmit | ❌ Todo | Run on real hardware, verify audio output (runbook added) |
| Collect real gesture samples | Nirmit | ❌ Todo | `python train_model.py --collect`, press 1-7 to set label, SPACE to record |
| Train model with real data | Both | ❌ Todo | After collecting: `python train_model.py --train` |
| Demo video recording | Nirmit | ❌ Todo | Screen record working prototype (demo tips added to README) |
| README + GIF finalization | Nirmit | ✅ In Progress | Added recording tips and GIF pipeline; finalize with example GIF later |

### How to Collect Real Gesture Data

```bash
# 1. Collect samples (webcam opens, press number keys to set label, SPACE to record)
python train_model.py --collect

# 2. Train from collected samples
python train_model.py --train --samples assets/samples.npz

# 3. Verify model works
python train_model.py --verify
```

---

## Critical Bugs (All Fixed ✅)

1. ~~`import mediapipe` inside while loop~~ → **Fixed**: moved to top-level
2. ~~Visualizer draws zeros~~ → **Fixed**: `update_from_theremin()` generates synthetic waveform from pitch/volume/waveform/filter_cutoff
3. ~~Camera disconnect crashes~~ → **Fixed**: 30-frame tolerance, graceful exit
4. ~~ONNX `FloatTensorType` deprecated~~ → **Fixed**: uses `skl2onnx.common.data_types.FloatTensorType`
5. ~~ONNX inference fails on zipmap output~~ → **Fixed**: handles both raw prob array and zipmap dict
6. ~~Model labels don't match training~~ → **Fixed**: labels embedded in ONNX metadata, read back on load

---

## File Manifest

```
ThereSyn/
├── main.py                    # Entry point (fixed)
├── config.py                  # All constants
├── train_model.py             # ML training pipeline (NEW)
├── requirements.txt
├── BUILD_PLAN.md
├── README.md
├── dsp/
│   ├── oscillator.py          # 4 waveforms, band-limited
│   ├── envelope.py            # ADSR
│   └── filter.py              # Lowpass
├── engine/
│   ├── audio_engine.py        # Threaded synthesis
│   ├── midi_output.py         # Pitch bend + CC
│   └── latency_profiler.py    # Per-stage profiling
├── gesture/
│   ├── gesture_classifier.py  # ONNX inference (FIXED)
│   ├── pinch_detector.py      # Pinch engagement
│   └── theremin_mapper.py     # X→pitch, Y→volume, deadzones
├── ui/
│   ├── theremin_ui.py         # Overlay panel
│   └── visualizer.py          # Waveform + spectrum (FIXED)
├── utils/
│   ├── smoothing.py           # EMA landmark smoother
│   ├── debounce.py            # Debouncer
│   └── logger.py              # Logger
├── vision/
│   ├── camera.py              # OpenCV capture
│   └── hand_tracker.py        # MediaPipe hands
└── tests/
    ├── test_phase1.py          # 6 tests
    ├── test_phase2.py          # 7 tests
    ├── test_phase3.py          # 4 tests
    ├── test_audio_engine_headless.py
    ├── test_gesture_classifier.py
    ├── test_latency_profiler.py
    ├── test_midi.py
    ├── test_smoothing.py
    └── test_theremin_mapper.py
```
