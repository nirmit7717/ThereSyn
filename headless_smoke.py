"""headless_smoke.py — Headless smoke-run for ThereSyn.

Simulates hand landmarks and runs the theremin mapping + audio engine in headless mode
to validate integration without webcam/audio hardware.
"""

import time
import math
from engine.audio_engine import AudioEngine
from gesture.theremin_mapper import ThereminMapper


def run_headless_smoke(duration_s: float = 1.0, steps: int = 40) -> bool:
    """Run a short headless simulation of the theremin pipeline.

    Returns True on success (no exceptions), False otherwise.
    """
    audio = AudioEngine(enable_audio=False)
    mapper = ThereminMapper()

    prev_engaged = False
    try:
        for i in range(steps):
            t = i / float(max(1, steps - 1))
            # Simulate wrist X sweeping back and forth
            wrist_x = 0.5 + 0.4 * math.sin(2 * math.pi * (t * 2.0))
            wrist_y = 0.5

            # Create simple landmarks: wrist at index 0, finger tips spread around
            landmarks = [(wrist_x, wrist_y, 0.0)]
            for j in range(1, 21):
                # scatter other landmarks around the wrist position
                dx = (j % 5 - 2) * 0.02
                dy = ((j // 5) - 2) * 0.015
                landmarks.append((wrist_x + dx, wrist_y + dy, 0.0))

            # Engage every other half-cycle to exercise engage/disengage paths
            engaged = (int(t * 4) % 2) == 0

            out = mapper.map_hand(landmarks, engaged)
            freq = out["pitch"]
            vol = out["volume"]
            cutoff = out["filter_cutoff"]

            # Pass to audio engine (headless — no actual sound)
            audio.update_theremin(freq, vol, filter_cutoff=cutoff)
            if engaged and not prev_engaged:
                audio.theremin_engage()
            if not engaged and prev_engaged:
                audio.theremin_disengage()

            prev_engaged = engaged
            # small sleep to simulate frame pacing
            time.sleep(duration_s / float(steps))

        # clean shutdown
        audio.quit()
        return True
    except Exception as e:
        try:
            audio.quit()
        except Exception:
            pass
        raise


if __name__ == '__main__':
    ok = run_headless_smoke(duration_s=1.0, steps=40)
    print('Headless smoke-run OK' if ok else 'Headless smoke-run FAILED')
