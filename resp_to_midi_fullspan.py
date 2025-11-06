#!/usr/bin/env python3
"""
Go Direct Respiration Belt → MIDI CC (0–127 full-span per breath)

- Creates a virtual MIDI port. Select it in your DAW and map to CC7 (Volume) or any CC.
- Uses EMA-smoothed force signal (sensor 1).
- Detects breath boundaries (trough→trough) from the EMA slope.
- On each breath boundary, sets in_min/in_max to the extremes of the *previous* breath,
  so the next breath sweeps ~0–127 linearly (no padding, no end-damping).
- Shows live plot of raw, EMA, and MIDI CC.

Requires: gdx, mido, python-rtmidi, matplotlib, numpy
"""

import time
from collections import deque
import numpy as np
import mido
from gdx import gdx
import matplotlib.pyplot as plt

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"
MIDI_CHANNEL       = 1          # 1–16
MIDI_CC            = 7          # CC number to send
READ_HZ            = 100        # sensor sample rate (Hz)
SMOOTH_ALPHA       = 0.3       # EMA: ema = α*ema + (1-α)*raw (α→1 = heavier smoothing)
MIN_SEND_STEP      = 0          # only send CC if change ≥ this amount
INITIAL_CALIB_SECS = 10         # quick warm-up to seed EMA and plot
SENSOR_CHANNELS    = [1]        # we only need force; you can add 2 but it's unused
PLOT_WINDOW_SECS   = 10

# Peak/trough detection
DERIV_EPS          = 1e-5       # derivative deadband (units of sensor value)
REFRACTORY_MS      = 250        # ignore new extrema for this time after we just detected one
MIN_BREATH_SPAN    = 1e-3       # minimum span (max-min) to accept a "breath" (avoid recal on micro noise)
BPM_ROLLING        = 6          # average last N breath periods for BPM

# ==========================

# ---- MIDI setup ----
mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")

def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    if in_max is None or in_min is None or in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return int(round(out_min + t * (out_max - out_min)))

class BreathPerCycleNormalizer:
    """
    Detect troughs from EMA slope sign change (− → +) with a refractory period.
    On each detected trough, we finalize the previous cycle extremes (min/max)
    and set them as the scaling range for the NEXT breath.
    """
    def __init__(self, fs, deriv_eps=1e-5, refractory_ms=250, min_span=1e-3, bpm_window=6):
        self.fs = fs
        self.deriv_eps = deriv_eps
        self.refractory = int(round(refractory_ms * fs / 1000.0))
        self.cool = 0

        self.prev_x = None
        self.prev_sign = 0

        # Track extrema over current trough→...→(next trough)
        self.cycle_min = float('inf')
        self.cycle_max = float('-inf')

        # Published range
        self.in_min = None
        self.in_max = None

        # BPM tracking
        self.last_trough_time = None
        self.breath_durations = deque(maxlen=bpm_window)
        self.total_breaths = 0

    def sign(self, v):
        if v > self.deriv_eps: return 1
        if v < -self.deriv_eps: return -1
        return 0

    def update(self, x, t):
        """Update with a new EMA sample at time t (seconds)."""
        # update running extrema
        self.cycle_min = min(self.cycle_min, x)
        self.cycle_max = max(self.cycle_max, x)

        if self.prev_x is None:
            self.prev_x = x
            return False  # no event yet

        d = x - self.prev_x
        s = self.sign(d)

        event = False
        if self.cool > 0:
            self.cool -= 1
        else:
            # detect trough: going down then up (− → +)
            if self.prev_sign < 0 and s > 0:
                span = self.cycle_max - self.cycle_min
                if span >= self.min_span:
                    # publish range = last cycle extremes
                    self.in_min, self.in_max = self.cycle_min, self.cycle_max
                    self.total_breaths += 1
                    if self.last_trough_time is not None:
                        self.breath_durations.append(t - self.last_trough_time)
                    self.last_trough_time = t
                    event = True
                # reset for next cycle (start new tracking from current point)
                self.cycle_min = x
                self.cycle_max = x
                self.cool = self.refractory

        self.prev_sign = s
        self.prev_x = x
        return event

    @property
    def min_span(self):
        return MIN_BREATH_SPAN

    def get_range(self):
        return self.in_min, self.in_max

    def get_bpm(self):
        if len(self.breath_durations) == 0:
            return None
        period = np.median(self.breath_durations)  # robust
        if period <= 0:
            return None
        return 60.0 / period

class LivePlotter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.raw_vals = deque(maxlen=window_size)
        self.smooth_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Respiration Belt → MIDI (Breath-Adaptive Calibration)')

        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)

        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC (linear)')
        self.ax2.set_ylabel('MIDI CC (0-127)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylim(-5, 132)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        self.info_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        self.start_time = time.time()
        plt.ion(); plt.show()

    def add(self, raw, ema, midi):
        t = time.time() - self.start_time
        self.times.append(t)
        self.raw_vals.append(raw)
        self.smooth_vals.append(ema)
        self.midi_vals.append(midi)

    def update(self, info_text=None):
        if not self.times:
            return
        times = list(self.times)
        self.line_raw.set_data(times, list(self.raw_vals))
        self.line_smooth.set_data(times, list(self.smooth_vals))
        self.ax1.relim(); self.ax1.autoscale_view()
        self.ax1.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)

        self.line_midi.set_data(times, list(self.midi_vals))
        self.ax2.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)

        if info_text:
            self.info_text.set_text(info_text)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def connect_device():
    g = gdx.gdx()
    print("\nConnecting to Go Direct Respiration Belt...")
    try:
        g.open(connection='ble')
        time.sleep(0.8)
    except Exception:
        pass
    if not getattr(g, "devices", None):
        try:
            g.open(connection='usb')
            time.sleep(0.5)
        except Exception as e:
            print("Could not connect via BLE or USB:", e)
            return None
    try:
        g.select_sensors(SENSOR_CHANNELS)
    except Exception:
        g.select_sensors([1])
    try:
        g.start(period=int(1000 / READ_HZ))
    except Exception as e:
        print("Failed to start streaming:", e)
        return None
    return g

def main():
    g = connect_device()
    if g is None:
        print("❌ No device. Exiting.")
        return

    # --- quick warmup to seed EMA ---
    print(f"Warm-up {INITIAL_CALIB_SECS}s...")
    t0 = time.time()
    ema = None
    warm_raw = []
    while time.time() - t0 < INITIAL_CALIB_SECS:
        data = g.read()
        if not data: continue
        raw = data[0]
        ema = raw if ema is None else (SMOOTH_ALPHA * ema + (1 - SMOOTH_ALPHA) * raw)
        warm_raw.append((raw, ema))
        time.sleep(1.0 / READ_HZ)
    print("Warm-up done.")

    # --- plotter & normalizer ---
    plotter = LivePlotter(window_size=int(PLOT_WINDOW_SECS * READ_HZ))
    norm = BreathPerCycleNormalizer(
        fs=READ_HZ,
        deriv_eps=DERIV_EPS,
        refractory_ms=REFRACTORY_MS,
        min_span=MIN_BREATH_SPAN,
        bpm_window=BPM_ROLLING
    )

    # seed plot with warm data (optional)
    for raw, e in warm_raw[-int(READ_HZ):]:
        norm.update(e, time.time())  # just to initialize internal state
        plotter.add(raw, e, 0)

    # current range
    in_min, in_max = None, None
    last_sent = None
    frame = 0

    print("\nStreaming. Press Ctrl+C to stop.")
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]
            ema = raw if ema is None else (SMOOTH_ALPHA * ema + (1 - SMOOTH_ALPHA) * raw)

            # Update per-breath detector; publish range on trough
            event = norm.update(ema, time.time())
            if event:
                in_min, in_max = norm.get_range()
                if in_min is not None and in_max is not None:
                    print(f"New breath range → [{in_min:.5f}, {in_max:.5f}] | "
                          f"BPM: {(norm.get_bpm() or 0):.1f} | Total breaths: {norm.total_breaths}")

            # If we still don't have a range (first cycle), fall back to small band around current EMA
            if in_min is None or in_max is None or in_max <= in_min:
                in_min = ema - 1e-3
                in_max = ema + 1e-3

            # Map EMA → 0..127
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)

            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                _midi_out.send(mido.Message('control_change',
                                            channel=MIDI_CHANNEL - 1,
                                            control=MIDI_CC,
                                            value=cc_val))
                last_sent = cc_val

            # Plot
            plotter.add(raw, ema, cc_val)
            frame += 1
            if frame % 10 == 0:
                bpm = norm.get_bpm()
                info = (
                    f"Breathing Rate: {bpm:.1f} BPM" if bpm else "Breathing Rate: detecting..."
                )
                info += (f"\nTotal breaths: {norm.total_breaths}"
                         f"\nRange: [{in_min:.3f}, {in_max:.3f}]")
                plotter.update(info)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try: _midi_out.close()
        except Exception: pass
        try:
            g.stop(); g.close()
        except Exception: pass
        plt.close('all')
        print("Goodbye.")

if __name__ == "__main__":
    main()
