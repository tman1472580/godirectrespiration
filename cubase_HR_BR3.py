#!/usr/bin/env python3
"""
Go Direct Respiration Belt → MIDI CC (0–127 full-span per breath) + Polar H10 Heart Rate Monitor

- Creates a virtual MIDI port. Select it in your DAW and map to CC7 (Volume) or any CC.
- Uses EMA-smoothed force signal (sensor 1).
- Detects breath boundaries (trough→trough) from the EMA slope.
- Calculates breathing rate from detected breath cycles (fallback if sensor 2 fails).
- Monitors heart rate via Polar H10 Bluetooth device.
- Shows start and end metrics for both heart rate and breathing rate.
- Shows live plot of raw, EMA, MIDI CC, and heart rate.
- Interactive sliders for sigmoid strength and smoothing alpha.

Requires: gdx, mido, python-rtmidi, matplotlib, numpy, bleak
"""

import time
import asyncio
import threading
from collections import deque
import numpy as np
import mido
from gdx import gdx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from bleak import BleakScanner
from PolarH10 import PolarH10

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"
MIDI_CHANNEL       = 1          # 1–16
MIDI_CC            = 7          # CC number to send
READ_HZ            = 100        # sensor sample rate (Hz)
SMOOTH_ALPHA       = 0.3       # EMA: ema = α*ema + (1-α)*raw (α→1 = heavier smoothing)
MIN_SEND_STEP      = 0          # only send CC if change ≥ this amount
INITIAL_CALIB_SECS = 10         # quick warm-up to seed EMA and plot
SENSOR_CHANNELS    = [1, 2]     # 1=force for MIDI, 2=breathing rate from device (may not work)
PLOT_WINDOW_SECS   = 10

# Peak/trough detection
DERIV_EPS          = 1e-5       # derivative deadband (units of sensor value)
REFRACTORY_MS      = 250        # ignore new extrema for this time after we just detected one
MIN_BREATH_SPAN    = 1e-3       # minimum span (max-min) to accept a "breath" (avoid recal on micro noise)
BPM_ROLLING        = 6          # average last N breath periods for BPM

# Polar H10 settings
POLAR_DEVICE_NAME  = "Polar H10"  # Device name prefix to search for
HR_WARMUP_SECS     = 5            # Time to collect initial HR readings

# Breathing rate sensor settings
BR_WARMUP_BREATHS  = 3            # Wait for this many breaths before using sensor BR
BR_VALID_RANGE     = (2.0, 60.0)  # Valid breathing rate range in BrPM

# Sigmoid settings
SIGMOID_STRENGTH   = 4.5          # Initial sigmoid strength

# ==========================

# ---- MIDI setup ----
mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")

# Global parameters that can be modified by sliders
class GlobalParams:
    def __init__(self):
        self.smooth_alpha = SMOOTH_ALPHA
        self.sigmoid_strength = SIGMOID_STRENGTH
        self.lock = threading.Lock()
    
    def set_smooth_alpha(self, value):
        with self.lock:
            self.smooth_alpha = value
    
    def get_smooth_alpha(self):
        with self.lock:
            return self.smooth_alpha
    
    def set_sigmoid_strength(self, value):
        with self.lock:
            self.sigmoid_strength = value
    
    def get_sigmoid_strength(self):
        with self.lock:
            return self.sigmoid_strength

global_params = GlobalParams()

def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    if in_max is None or in_min is None or in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    # Apply sigmoid transformation with adjustable strength
    import math
    strength = global_params.get_sigmoid_strength()
    t = 1 / (1 + math.exp(-strength * (t - 0.5)))
    return int(round(out_min + t * (out_max - out_min)))

class BreathingRateMonitor:
    """Track breathing rate from device sensor 2 OR calculated from breath cycles."""
    def __init__(self):
        self.br_readings = deque(maxlen=20)
        self.start_br = None
        self.start_time = None
        self.valid_reading_count = 0
        self.ready = False
        self.sensor_working = False
        self.debug_count = 0
        
    def update_from_sensor(self, br_value):
        """Update with new breathing rate from sensor 2."""
        # Debug: print first 10 values to see what we're getting
        if self.debug_count < 10:
            print(f"Debug BR sensor value: {br_value} (type: {type(br_value).__name__})")
            self.debug_count += 1
        
        # Check if value is valid (not nan, not None, in reasonable range)
        if br_value is not None and not np.isnan(br_value) and BR_VALID_RANGE[0] <= br_value <= BR_VALID_RANGE[1]:
            self.br_readings.append(br_value)
            self.valid_reading_count += 1
            self.sensor_working = True
            
            # Mark as ready after getting enough valid readings
            if not self.ready and self.valid_reading_count >= 10:
                self.ready = True
                print(f"✓ Breathing rate sensor stabilized (using sensor data)")
            
            # Set start BR if not set and we have enough stable readings
            if self.start_br is None and len(self.br_readings) >= 10:
                avg_br = np.median(list(self.br_readings))
                if BR_VALID_RANGE[0] <= avg_br <= BR_VALID_RANGE[1]:
                    self.start_br = avg_br
                    self.start_time = time.time()
            
            return br_value
        return None
    
    def update_from_cycles(self, breath_durations):
        """Calculate breathing rate from detected breath cycles (fallback method)."""
        if len(breath_durations) >= 3:
            # Calculate BPM from breath periods
            median_period = np.median(list(breath_durations))
            if median_period > 0:
                br = 60.0 / median_period
                if BR_VALID_RANGE[0] <= br <= BR_VALID_RANGE[1]:
                    self.br_readings.append(br)
                    
                    # Mark as ready using calculated method
                    if not self.ready and len(self.br_readings) >= 5:
                        self.ready = True
                        if self.debug_count >= 10 and not self.sensor_working:
                            print(f"✓ Breathing rate calculated from breath cycles (sensor data unavailable)")
                    
                    # Set start BR
                    if self.start_br is None and len(self.br_readings) >= 5:
                        avg_br = np.median(list(self.br_readings))
                        if BR_VALID_RANGE[0] <= avg_br <= BR_VALID_RANGE[1]:
                            self.start_br = avg_br
                            self.start_time = time.time()
                    
                    return br
        return None
    
    def get_current_br(self):
        """Get smoothed current breathing rate."""
        if self.ready and len(self.br_readings) >= 3:
            return np.median(list(self.br_readings))
        return None
    
    def get_start_br(self):
        """Get the starting breathing rate."""
        return self.start_br
    
    def is_ready(self):
        """Check if we have stable breathing rate data."""
        return self.ready

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

    def get_breath_durations(self):
        """Get recent breath durations for BR calculation."""
        return self.breath_durations

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
        self.hr_vals = deque(maxlen=window_size)

        # Create figure with extra space for sliders
        self.fig = plt.figure(figsize=(12, 11))
        
        # Create grid for plots and sliders
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 2, 2, 2, 0.8], hspace=0.4, top=0.95, bottom=0.08)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax4 = self.fig.add_subplot(gs[3])
        
        self.fig.suptitle('Respiration Belt + Heart Rate → MIDI (Interactive Controls)', fontsize=14, fontweight='bold')

        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)

        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC (with sigmoid)')
        self.ax2.set_ylabel('MIDI CC (0-127)')
        self.ax2.set_ylim(-5, 132)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)

        self.line_hr, = self.ax3.plot([], [], 'm-', linewidth=2, label='Heart Rate')
        self.ax3.set_ylabel('Heart Rate (BPM)')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylim(40, 200)
        self.ax3.legend(loc='upper right')
        self.ax3.grid(True, alpha=0.3)

        # Sigmoid function visualization
        self.sigmoid_x = np.linspace(0, 1, 200)
        self.line_sigmoid, = self.ax4.plot([], [], 'orange', linewidth=2, label='Sigmoid Transform')
        self.line_linear, = self.ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Linear')
        self.ax4.set_xlabel('Input (normalized 0-1)')
        self.ax4.set_ylabel('Output (0-1)')
        self.ax4.set_xlim(0, 1)
        self.ax4.set_ylim(0, 1)
        self.ax4.legend(loc='upper left')
        self.ax4.grid(True, alpha=0.3)
        self.ax4.set_title('Sigmoid Transform Curve')

        self.info_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # Create sliders
        ax_sigmoid = plt.axes([0.15, 0.04, 0.35, 0.02])
        ax_alpha = plt.axes([0.65, 0.04, 0.25, 0.02])
        
        self.slider_sigmoid = Slider(
            ax_sigmoid, 'Sigmoid\nStrength', 0.5, 10.0, 
            valinit=SIGMOID_STRENGTH, valstep=0.1
        )
        self.slider_alpha = Slider(
            ax_alpha, 'Smooth α', 0.01, 0.99, 
            valinit=SMOOTH_ALPHA, valstep=0.01
        )
        
        # Connect slider events
        self.slider_sigmoid.on_changed(self.update_sigmoid)
        self.slider_alpha.on_changed(self.update_alpha)

        self.start_time = time.time()
        self.update_sigmoid_curve()
        plt.ion(); plt.show()

    def update_sigmoid(self, val):
        """Callback for sigmoid strength slider."""
        global_params.set_sigmoid_strength(val)
        self.update_sigmoid_curve()

    def update_alpha(self, val):
        """Callback for smooth alpha slider."""
        global_params.set_smooth_alpha(val)

    def update_sigmoid_curve(self):
        """Update the sigmoid curve visualization."""
        import math
        strength = global_params.get_sigmoid_strength()
        sigmoid_y = 1 / (1 + np.exp(-strength * (self.sigmoid_x - 0.5)))
        self.line_sigmoid.set_data(self.sigmoid_x, sigmoid_y)

    def add(self, raw, ema, midi, hr=None):
        t = time.time() - self.start_time
        self.times.append(t)
        self.raw_vals.append(raw)
        self.smooth_vals.append(ema)
        self.midi_vals.append(midi)
        if hr is not None:
            self.hr_vals.append(hr)

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

        if self.hr_vals:
            hr_times = times[-len(self.hr_vals):]
            self.line_hr.set_data(hr_times, list(self.hr_vals))
            self.ax3.relim(); self.ax3.autoscale_view(scaley=True, scalex=False)
            self.ax3.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)

        # Update sigmoid curve
        self.update_sigmoid_curve()

        if info_text:
            # Add slider values to info text
            strength = global_params.get_sigmoid_strength()
            alpha = global_params.get_smooth_alpha()
            info_text += f"\nSigmoid: {strength:.1f} | Alpha: {alpha:.2f}"
            self.info_text.set_text(info_text)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def connect_device():
    """Connect to Go Direct device synchronously."""
    g = gdx.gdx()
    print("\nConnecting to Go Direct Respiration Belt...")
    
    # Try BLE first
    try:
        g.open(connection='ble')
        time.sleep(0.8)
    except Exception as e:
        print(f"BLE connection failed: {e}")
    
    # If BLE didn't work, try USB
    if not getattr(g, "devices", None):
        try:
            print("Trying USB connection...")
            g.open(connection='usb')
            time.sleep(0.5)
        except Exception as e:
            print(f"USB connection failed: {e}")
            print("Could not connect via BLE or USB")
            return None
    
    # Select sensors - handle gracefully if sensor 2 fails
    try:
        print(f"Selecting sensors: {SENSOR_CHANNELS}")
        g.select_sensors(SENSOR_CHANNELS)
        print(f"✓ Successfully selected sensors: {SENSOR_CHANNELS}")
    except Exception as e:
        print(f"Warning: Could not select all sensors: {e}")
        print("Trying sensor 1 (Force) only...")
        try:
            g.select_sensors([1])
            print("✓ Selected sensor 1 (Force) - will calculate breathing rate from cycles")
        except Exception as e2:
            print(f"Failed to select any sensors: {e2}")
            return None
    
    # Start streaming
    try:
        g.start(period=int(1000 / READ_HZ))
        print(f"✓ Started streaming at {READ_HZ} Hz")
    except Exception as e:
        print(f"Failed to start streaming: {e}")
        return None
    
    return g

class HeartRateMonitor:
    """Track heart rate metrics with thread-safe access."""
    def __init__(self):
        self.hr_readings = deque(maxlen=10)
        self.start_hr = None
        self.start_time = None
        self.polar = None
        self.lock = threading.Lock()
        
    def update(self):
        """Get latest HR from Polar H10."""
        if not self.polar or not self.polar.ibi_stream_values:
            return None
            
        with self.lock:
            # Calculate instantaneous HR from most recent IBI
            recent_ibis = self.polar.ibi_stream_values[-10:]  # Last 10 IBIs
            if recent_ibis:
                avg_ibi_ms = np.mean(recent_ibis)
                hr = 60000.0 / avg_ibi_ms  # Convert ms to BPM
                self.hr_readings.append(hr)
                
                # Set start HR if not set and we have enough stable readings
                if self.start_hr is None and len(self.hr_readings) >= 5:
                    self.start_hr = np.mean(list(self.hr_readings))
                    self.start_time = time.time()
                    
                return hr
        return None
    
    def get_current_hr(self):
        """Get smoothed current heart rate."""
        with self.lock:
            if self.hr_readings:
                return np.mean(list(self.hr_readings))
        return None
    
    def get_start_hr(self):
        """Get the starting heart rate."""
        with self.lock:
            return self.start_hr

async def setup_polar_in_thread(hr_monitor):
    """Setup Polar H10 in background thread."""
    try:
        print(f"\nScanning for {POLAR_DEVICE_NAME} device...")
        devices = await BleakScanner.discover(timeout=10.0)
        polar_device = None
        
        for device in devices:
            if device.name and POLAR_DEVICE_NAME in device.name:
                print(f"Found {device.name} {device.address}")
                polar_device = device
                break
        
        if not polar_device:
            print(f"❌ No {POLAR_DEVICE_NAME} device found.")
            return None
        
        polar = PolarH10(polar_device)
        await polar.connect()
        await polar.get_device_info()
        await polar.print_device_info()
        await polar.start_hr_stream()
        
        with hr_monitor.lock:
            hr_monitor.polar = polar
        
        print(f"\nCollecting initial heart rate for {HR_WARMUP_SECS}s...")
        t0 = time.time()
        while time.time() - t0 < HR_WARMUP_SECS:
            hr_monitor.update()
            await asyncio.sleep(0.1)
        
        print("✓ Heart rate monitoring ready")
        
        # Keep the async loop running to maintain connection
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Polar H10 error: {e}")
        return None

def start_polar_thread(hr_monitor):
    """Start Polar H10 in a separate thread with its own event loop."""
    def run_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(setup_polar_in_thread(hr_monitor))
        except Exception as e:
            print(f"Polar thread error: {e}")
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_async_loop, daemon=True)
    thread.start()
    return thread

def main():
    # Connect to respiration belt
    g = connect_device()
    if g is None:
        print("❌ No device. Exiting.")
        return

    # Setup heart rate monitor in background thread
    hr_monitor = HeartRateMonitor()
    polar_thread = start_polar_thread(hr_monitor)
    
    # Setup breathing rate monitor
    br_monitor = BreathingRateMonitor()
    
    # Give Polar time to connect and warm up
    time.sleep(HR_WARMUP_SECS + 2)

    # --- quick warmup to seed EMA ---
    print(f"\nWarm-up {INITIAL_CALIB_SECS}s...")
    t0 = time.time()
    ema = None
    warm_raw = []
    
    while time.time() - t0 < INITIAL_CALIB_SECS:
        data = g.read()
        if not data:
            time.sleep(1.0 / READ_HZ)
            continue
        
        raw = data[0]  # Force sensor (always available)
        
        # Try to get breathing rate from sensor 2 if available
        br_value = None
        if len(data) > 1:
            br_value = data[1]
            # Update breathing rate monitor with sensor data
            if br_value is not None:
                br_monitor.update_from_sensor(br_value)
        
        # Use current smooth alpha from global params
        alpha = global_params.get_smooth_alpha()
        ema = raw if ema is None else (alpha * ema + (1 - alpha) * raw)
        warm_raw.append((raw, ema))
        hr_monitor.update()
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

    # seed plot with warm data
    for raw, e in warm_raw[-int(READ_HZ):]:
        norm.update(e, time.time())
        plotter.add(raw, e, 0, hr_monitor.get_current_hr())

    # Display starting metrics
    start_hr = hr_monitor.get_start_hr()
    
    print("\n" + "="*60)
    print("SESSION START METRICS")
    print("="*60)
    if start_hr:
        print(f"Heart Rate:     {start_hr:.1f} BPM")
    else:
        print(f"Heart Rate:     Detecting...")
    print(f"Breathing Rate: Detecting...")
    print("="*60 + "\n")

    # current range
    in_min, in_max = None, None
    last_sent = None
    frame = 0
    start_breath_rate = None

    print("Streaming. Use sliders to adjust parameters. Press Ctrl+C to stop.\n")
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]  # Force sensor
            
            # Try to get breathing rate from sensor 2
            br_value = None
            if len(data) > 1:
                br_value = data[1]
            
            # Use current smooth alpha from global params
            alpha = global_params.get_smooth_alpha()
            ema = raw if ema is None else (alpha * ema + (1 - alpha) * raw)

            # Update heart rate
            current_hr = hr_monitor.update()
            
            # Update breathing rate from device sensor (if available)
            current_br = None
            if br_value is not None:
                current_br = br_monitor.update_from_sensor(br_value)
            
            # Update per-breath detector for range calibration AND breathing rate calculation
            event = norm.update(ema, time.time())
            
            # If sensor BR not working, calculate from breath cycles
            if not br_monitor.sensor_working:
                breath_durations = norm.get_breath_durations()
                if len(breath_durations) >= 3:
                    calculated_br = br_monitor.update_from_cycles(breath_durations)
                    if calculated_br and current_br is None:
                        current_br = calculated_br
            
            # Capture start breathing rate once we have stable data
            if start_breath_rate is None and br_monitor.is_ready():
                start_breath_rate = br_monitor.get_start_br()
                if start_breath_rate:
                    print(f"\n✓ Start breathing rate captured: {start_breath_rate:.1f} BrPM\n")

            if event:
                in_min, in_max = norm.get_range()
                
                if in_min is not None and in_max is not None:
                    br_display = br_monitor.get_current_br()
                    hr_display = hr_monitor.get_current_hr()
                    print(f"New breath range → [{in_min:.5f}, {in_max:.5f}] | "
                          f"Breathing: {(br_display or 0):.1f} BrPM | "
                          f"Heart: {(hr_display or 0):.1f} BPM | "
                          f"Total breaths: {norm.total_breaths}")

            # If we still don't have a range (first cycle), fall back to small band around current EMA
            if in_min is None or in_max is None or in_max <= in_min:
                in_min = ema - 1e-3
                in_max = ema + 1e-3

            # Map EMA → 0..127 (using current sigmoid strength from global params)
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)

            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                _midi_out.send(mido.Message('control_change',
                                            channel=MIDI_CHANNEL - 1,
                                            control=MIDI_CC,
                                            value=cc_val))
                last_sent = cc_val

            # Plot
            plotter.add(raw, ema, cc_val, current_hr)
            frame += 1
            if frame % 10 == 0:
                current_br_display = br_monitor.get_current_br()
                info = ""
                if current_br_display:
                    info = f"Breathing Rate: {current_br_display:.1f} BrPM"
                else:
                    info = "Breathing Rate: stabilizing..."
                
                info += f"\nTotal breaths: {norm.total_breaths}"
                info += f"\nRange: [{in_min:.3f}, {in_max:.3f}]"
                
                if current_hr:
                    info += f"\nHeart Rate: {current_hr:.1f} BPM"
                
                plotter.update(info)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\n\nStopping…")
    finally:
        # Display end metrics
        end_breath_rate = br_monitor.get_current_br()
        end_hr = hr_monitor.get_current_hr()
        
        print("\n" + "="*60)
        print("SESSION END METRICS")
        print("="*60)
        if end_hr:
            print(f"Heart Rate:     {end_hr:.1f} BPM")
        else:
            print(f"Heart Rate:     Not available")
        if end_breath_rate:
            print(f"Breathing Rate: {end_breath_rate:.1f} BrPM")
        else:
            print(f"Breathing Rate: Not available")
        print(f"Total Breaths:  {norm.total_breaths}")
        print("="*60)
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print("START:")
        if start_hr:
            print(f"  Heart Rate:     {start_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:     Not available")
        if start_breath_rate:
            print(f"  Breathing Rate: {start_breath_rate:.1f} BrPM")
        else:
            print(f"  Breathing Rate: Not available")
        
        print("\nEND:")
        if end_hr:
            print(f"  Heart Rate:     {end_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:     Not available")
        if end_breath_rate:
            print(f"  Breathing Rate: {end_breath_rate:.1f} BrPM")
        else:
            print(f"  Breathing Rate: Not available")
        
        if start_hr and end_hr:
            hr_change = end_hr - start_hr
            print(f"\nHeart Rate Change: {hr_change:+.1f} BPM")
        if start_breath_rate and end_breath_rate:
            br_change = end_breath_rate - start_breath_rate
            print(f"Breathing Rate Change: {br_change:+.1f} BrPM")
        elif not start_breath_rate:
            print(f"\nBreathing Rate Change: Not available (insufficient data)")
        print("="*60 + "\n")
        
        # Cleanup
        try: 
            _midi_out.close()
        except Exception: 
            pass
        try:
            g.stop()
            g.close()
        except Exception: 
            pass
        if hr_monitor.polar:
            print("Disconnecting Polar H10...")
        plt.close('all')
        print("Goodbye.")

if __name__ == "__main__":
    main()