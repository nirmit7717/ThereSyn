"""
ui/visualizer.py — Real-time audio waveform and spectrum display.

Generates a visual representation of the current theremin output
using synthetic waveforms based on current pitch/volume/waveform.
"""

import math
import numpy as np
import pygame


class AudioVisualizer:
    def __init__(self, width: int = 640, height: int = 80, sample_rate: int = 44100):
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self._waveform = np.zeros(width)
        self._spectrum = np.zeros(width // 2)
        self._phase = 0.0  # running phase for continuous waveform display

    def update_from_theremin(self, frequency: float, volume: float,
                             waveform: str = "sine", filter_cutoff: float = 1.0):
        """
        Generate a synthetic waveform preview from current theremin state.

        This creates a short visual snippet of what the waveform looks like
        at the current frequency/volume, for display purposes only.
        """
        if volume < 0.01:
            # Decay to silence
            self._waveform *= 0.85
            self._spectrum *= 0.85
            return

        # Generate one full display window of the waveform
        # Show ~3 cycles of the waveform regardless of frequency
        t = np.linspace(0, 1.0, self.width, endpoint=False)

        if waveform == "sine":
            wave = np.sin(2 * np.pi * 3 * t)  # 3 visible cycles
        elif waveform == "sawtooth":
            wave = 2 * (3 * t - np.floor(3 * t + 0.5))  # band-limited visual
        elif waveform == "square":
            wave = np.sign(np.sin(2 * np.pi * 3 * t))
        elif waveform == "triangle":
            wave = 2 * np.abs(2 * (3 * t - np.floor(3 * t + 0.5))) - 1
        else:
            wave = np.sin(2 * np.pi * 3 * t)

        # Apply volume
        wave = wave * volume

        # Apply filter effect visually (round corners when cutoff is low)
        if filter_cutoff < 0.5:
            # Simple moving average to simulate filtered look
            kernel_size = max(3, int((1.0 - filter_cutoff) * 20))
            kernel = np.ones(kernel_size) / kernel_size
            wave = np.convolve(wave, kernel, mode='same')

        # Smooth transition from previous frame
        self._waveform = self._waveform * 0.3 + wave * 0.7

        # Generate spectrum visualization from frequency
        self._update_spectrum_from_freq(frequency, volume, filter_cutoff)

    def _update_spectrum_from_freq(self, frequency: float, volume: float,
                                    filter_cutoff: float = 1.0):
        """Generate a synthetic spectrum display from current theremin params."""
        n_bins = self.width // 2

        # Map frequency to bin position (log scale)
        # freq range: ~100Hz to ~2000Hz, mapped across display width
        freq_min, freq_max = 100, 2000
        if frequency < freq_min:
            frequency = freq_min
        if frequency > freq_max:
            frequency = freq_max

        # Log frequency position
        log_pos = (math.log2(frequency) - math.log2(freq_min)) / \
                  (math.log2(freq_max) - math.log2(freq_min))
        center_bin = int(log_pos * n_bins)

        # Create a Gaussian peak at the fundamental + harmonics
        spectrum = np.zeros(n_bins)
        sigma = max(2, n_bins * 0.02)  # narrow peak

        for harmonic, amplitude in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)]:
            h_bin = min(center_bin * harmonic, n_bins - 1)
            if h_bin < n_bins:
                x = np.arange(n_bins)
                peak = amplitude * np.exp(-0.5 * ((x - h_bin) / sigma) ** 2)
                # Apply filter cutoff: suppress higher harmonics when cutoff is low
                if harmonic > 1:
                    peak *= filter_cutoff ** (harmonic - 1)
                spectrum += peak

        # Normalize
        peak_val = np.max(spectrum)
        if peak_val > 0:
            spectrum = spectrum / peak_val

        spectrum *= volume

        # Smooth with previous frame
        self._spectrum = self._spectrum * 0.5 + spectrum * 0.5

    def update_waveform(self, data: np.ndarray | None):
        """Update waveform display with raw audio data (if available)."""
        if data is not None and len(data) > 0:
            indices = np.linspace(0, len(data) - 1, self.width, dtype=int)
            self._waveform = data[indices]
        else:
            self._waveform *= 0.9

    def draw_waveform(self, surface, x: int = 0, y: int = 0):
        """Draw waveform overlay."""
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 100))
        surface.blit(panel, (x, y))

        center_y = self.height // 2
        points = []
        for i in range(self.width):
            px = x + i
            py = y + center_y - int(self._waveform[i] * (self.height // 2 - 4))
            py = max(y + 2, min(y + self.height - 2, py))
            points.append((px, py))

        if len(points) > 1:
            max_amp = np.max(np.abs(self._waveform))
            if max_amp > 0.01:
                # Color intensity based on amplitude
                green = int(100 + 155 * min(1.0, max_amp))
                pygame.draw.lines(surface, (0, green, 100), False, points, 2)
            else:
                # Draw flat center line when silent
                pygame.draw.line(surface, (40, 80, 60), (x, y + center_y),
                                 (x + self.width, y + center_y), 1)

    def draw_spectrum(self, surface, x: int = 0, y: int = 0):
        """Draw frequency spectrum bars."""
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 100))
        surface.blit(panel, (x, y))

        n_bars = len(self._spectrum)
        bar_w = max(1, self.width // n_bars)
        for i, magnitude in enumerate(self._spectrum):
            bx = x + i * bar_w
            h = int(magnitude * (self.height - 4))
            if h > 1:
                color = (
                    min(255, int(100 + 155 * (i / n_bars))),
                    max(0, int(255 - 200 * (i / n_bars))),
                    100,
                )
                pygame.draw.rect(surface, color, (bx, y + self.height - h - 2, bar_w - 1, h))
