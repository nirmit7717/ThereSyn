# ThereSyn — Build Plan (2 People, ~2 Days)

## Overall Progress

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: Core Pipeline | ✅ Complete | All pass |
| Phase 2: UI + Filters | ⚠️ Mostly done | See below |
| Phase 3: ML + MIDI + Latency | ✅ Complete | All pass |
| Phase 4: Integration + Demo | ❌ Not started | — |

---

## Phase 1: Core Pipeline — ✅ COMPLETE

All modules implemented and tested by both contributors.

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Camera + HandTracker | Nirmit | ✅ Done | Robustness additions added |
| EMA Smoothing | Collaborator | ✅ Done | Tested with multi-hand, hand loss |
| Pinch Detector | Collaborator | ✅ Done | Edge detection (just_pinched, just_released) |
| Theremin Mapper | Both | ✅ Done | Deadzones, exponential pitch, filter cutoff |
| DSP Oscillator | Collaborator | ✅ Done | 4 waveforms, band-limited, freq accurate |
| ADSR Envelope | Collaborator | ✅ Done | Edge cases (short signals, single sample) |
| Audio Engine | Nirmit | ✅ Done | Threaded, DI for headless, error handling |
| Main Loop Wiring | Both | ✅ Done | ML octave control, null-safety on MIDI |

---

## Phase 2: UI + Polish + Filters — ⚠️ MOSTLY DONE

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| Theremin UI (freq, vol, pinch indicator) | Collaborator | ✅ Done | Draws panel, bars, engagement ring |
| Landmark overlay on frame | Nirmit | ✅ Done | Working in main.py |
| Lowpass Filter | Collaborator | ✅ Done | Sweep tested, dark < mid < bright |
| Additional waveforms (saw, square, tri) | Collaborator | ✅ Done | Band-limited anti-aliased |
| Waveform cycling (W key + indicator) | Both | ✅ Done | Working in main.py |
| Audio Visualizer (waveform overlay) | Collaborator | ⚠️ Partial | **Module exists but never fed audio data in main.py — always draws zeros** |

### Remaining for Phase 2:
- **[Nirmit]** Feed real audio data to `visualizer.update_waveform()` from audio engine, or capture pygame mixer output. Currently `draw_waveform()` renders but shows nothing because `update_waveform()` is never called in `main.py`.

---

## Phase 3: ML + MIDI + Latency — ✅ COMPLETE

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| MIDI output (pitch_bend + CC) | Both | ✅ Done | Configurable bend range via config |
| MIDI wiring in main loop | Nirmit | ✅ Done | Null-safety, note_on/note_off |
| Latency Profiler | Nirmit | ✅ Done | Reporting avg/max/p95 per stage |
| ML Data Collector | Collaborator | ✅ Done | Record, save, load samples |
| ML Gesture Classifier | Both | ✅ Done | ONNX inference, fallback mode |
| ML wiring in main loop | Nirmit | ✅ Done | Auto-engage, stop, octave up/down |

### Known Issues:
- **ONNX export** has a version incompatibility (`onnx.FloatTensorType` deprecated in newer onnx). Training works in sklearn, ONNX export needs `skl2onnx` update. For now, the classifier runs in fallback mode without a trained model.
- **ML model not trained** — `assets/gesture_classifier.onnx` doesn't exist yet. The classifier falls back gracefully (returns `None`). Needs real hand gesture data collection + training.

---

## Phase 4: Integration + Demo — ❌ NOT STARTED

| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| End-to-end testing with camera | Nirmit | ❌ Todo | Run on real hardware, verify audio output |
| Fix mediapipe import inside loop | Nirmit | ❌ Todo | `import mediapipe as mp` at line 116 inside while loop — move to top of file |
| Feed visualizer with real data | Nirmit | ❌ Todo | `visualizer.update_waveform()` never called in main.py |
| Train ML model with real gestures | Both | ❌ Todo | Collect samples from webcam, train, export ONNX |
| Fix ONNX export compatibility | Collaborator | ❌ Todo | Update `skl2onnx` usage for newer onnx API |
| Bug fixes + edge cases | Nirmit | ❌ Todo | Camera disconnect, hand loss recovery, audio glitches |
| Unit tests (merged suite) | Both | ❌ Todo | Nirmit's individual tests + collaborator's phase tests coexist, need unified `pytest` |
| Demo video recording | Nirmit | ❌ Todo | Screen record working prototype |
| README + GIF finalization | Nirmit | ❌ Todo | Add demo GIF, screenshots, installation video |

---

## Critical Bugs to Fix Before Demo

1. **`import mediapipe as mp` inside the while loop** (`main.py:116`) — should be at top of file. Importing every frame wastes ~5ms.
2. **Visualizer draws zeros** — `update_waveform()` is never called. Need to capture audio data from the engine or generate a synthetic waveform from current pitch.
3. **No graceful camera disconnect** — if webcam unplugged mid-run, `camera.read_frame()` returns `None` and the loop breaks silently.
4. **pygame.init() double-call** — `main()` calls `pygame.init()` then `AudioEngine.__init__()` also calls `pygame.mixer.init()`. Nirmit's branch fixed this (AudioEngine checks `get_init()` first), but `main()` still calls `pygame.init()` before `AudioEngine()` is created.

---

## Quick Wins (1-2 hours total)

These would make the biggest visual/functional impact:

1. Move `import mediapipe` to top of `main.py` (5 min)
2. Feed visualizer with synth waveform from current pitch (30 min)
3. Add FPS counter to UI overlay (10 min)
4. Add camera disconnect recovery (15 min)
5. Remove duplicate `pygame.init()` (5 min)
