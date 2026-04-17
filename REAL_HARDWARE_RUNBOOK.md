ThereSyn — Real Hardware Runbook

Purpose

This runbook documents steps to verify the ThereSyn pipeline on a real machine with webcam, audio output, and optional MIDI target (DAW).

Prerequisites

- Python 3.10+ (3.11 recommended)
- USB webcam or built-in camera
- Audio output device (speakers/headphones)
- Optional: MIDI destination (virtual loopback like loopMIDI on Windows, or physical MIDI device)
- Install requirements: pip install -r requirements.txt

Quick checklist

1. Ensure camera is available and not used by other apps.
2. Ensure audio device works. Set system default to target output.
3. (Optional) Create a virtual MIDI port if you want to route MIDI to a DAW.

Environment variables (optional)

- Set PYTHONUNBUFFERED=1 when running to see logs in real time.

Step-by-step verification

1) Basic smoke-run (headless, no devices)

    python headless_smoke.py

Expect: prints "Headless smoke-run OK" and exits. This verifies mapping + audio engine logic without hardware.

2) Run main with webcam + audio

    pip install -r requirements.txt
    python main.py

- On startup, accept webcam permissions. The UI shows webcam feed + overlay.
- Pinch (thumb + index) to engage sound. Move left/right for pitch, up/down for volume.

3) Collect gesture samples (if you plan to retrain)

    python train_model.py --collect

- Window will open showing live feed. Use number keys (1-N) to set label. Press SPACE to record a sample with the currently-set label. Press S to save to disk.
- Save to assets/samples.npz (default) or supply --samples path.

4) Train model locally

    python train_model.py --train --samples assets/samples.npz --output assets/gesture_classifier.onnx

Notes: ONNX export requires skl2onnx, onnx, onnxruntime. If missing, training will run but export may be skipped; install: pip install skl2onnx onnx onnxruntime

5) Verify ONNX model

    python train_model.py --verify --output assets/gesture_classifier.onnx

6) MIDI routing (optional)

- On Windows use loopMIDI to create a virtual port. In your DAW, select that port as MIDI input for an instrument.
- The project sends pitch_bend + CC; ensure the DAW's pitch-bend range matches config.MIDI_PITCH_BEND_RANGE_SEMITONES (default 2 semitones).

Troubleshooting

- Camera fails to open: close other apps (Zoom, browser tabs). If still fails, run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` to check.
- No sound: ensure system volume and app volume not muted. If using virtual MIDI, ensure DAW is armed to receive input.
- Mediapipe installation errors: follow mediapipe install guide; on Windows pip wheels may be platform-specific.

Demo recording suggestions (OBS)

- Use a display capture + webcam overlay capture, or record the application window directly.
- Record audio from the system output (monitor of the audio device) and the webcam mic if you want narration.
- Recommended OBS settings for GIF: record 10–20s high-framerate (30–60 fps) MP4, then convert to GIF using ffmpeg + gifsicle to optimize.

Example GIF pipeline (Linux/WSL or cross-platform via ffmpeg + gifsicle):

    ffmpeg -i demo.mp4 -vf "fps=15,scale=800:-1:flags=lanczos" -y demo-15fps.gif
    gifsicle -O3 --colors 256 demo-15fps.gif -o demo-optimized.gif

Notes on safety and performance

- Close all unnecessary programs to reduce CPU load. Real-time ML + audio synthesis is CPU sensitive.
- If you experience stutter, try lowering camera resolution in config.py or reduce synthesis settings.

Contact

If anything fails during these steps, capture logs (stdout/stderr) and a short screen recording and share them. Include OS, Python version, and the output of `pip freeze`.
