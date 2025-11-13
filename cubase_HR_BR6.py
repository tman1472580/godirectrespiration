#!/usr/bin/env python3
"""
Go Direct Respiration Belt → MIDI CC (0–127 full-span per breath) + Polar H10 Heart Rate Monitor

- Creates a virtual MIDI port. Select it in your DAW and map to CC7 (Volume) or any CC.
- Uses EMA-smoothed force signal (sensor 1).
- Detects breath boundaries (trough→trough) from the EMA slope.
- Calculates breathing rate EXCLUSIVELY from detected breath cycle durations.
- Monitors heart rate via Polar H10 Bluetooth device.
- Shows start and end metrics for both heart rate and breathing rate.
- Shows live plot of raw, EMA, MIDI CC, and heart rate.
- Interactive sliders for sigmoid strength, smoothing alpha, and calibration window.
- Visualizes calibration window on the belt readings plot.

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
from matplotlib.patches import Rectangle
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
SENSOR_CHANNELS    = [1, 2]     # Force sensor + breathing rate sensor
PLOT_WINDOW_SECS   = 10

# Peak/trough detection
DERIV_EPS          = 1e-5       # derivative deadband (units of sensor value)
REFRACTORY_MS      = 250        # ignore new extrema for this time after we just detected one
MIN_BREATH_SPAN    = 1e-3       # minimum span (max-min) to accept a "breath" (avoid recal on micro noise)
BPM_ROLLING        = 6          # average last N breath periods for BPM

# Polar H10 settings
POLAR_DEVICE_NAME  = "Polar H10"  # Device name prefix to search for
HR_WARMUP_SECS     = 5            # Time to collect initial HR readings

# Breathing rate settings - calculated from cycles only
BR_WARMUP_BREATHS  = 5            # Wait for this many breaths before reporting stable BR
BR_VALID_RANGE     = (2.0, 60.0)  # Valid breathing rate range in BrPM

# Sigmoid settings
SIGMOID_STRENGTH   = 4.5          # Initial sigmoid strength

# Calibration window settings
CALIB_WINDOW_BREATHS = 3          # Initial number of breaths to average for calibration

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
        self.calib_window_breaths = CALIB_WINDOW_BREATHS
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
    
    def set_calib_window_breaths(self, value):
        with self.lock:
            self.calib_window_breaths = int(value)
    
    def get_calib_window_breaths(self):
        with self.lock:
            return self.calib_window_breaths

global_params = GlobalParams()

def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    """
    Linear scaling (linearization) maps an input range to an output range.
    
    Why it's needed:
    1. The respiration belt outputs raw force values (arbitrary units, e.g., 2.3-4.7)
    2. MIDI CC values must be integers from 0-127
    3. We want min breath → 0, max breath → 127 for full dynamic range
    
    Without this:
    - Raw sensor values would be meaningless to MIDI
    - You'd lose resolution or have values outside valid range
    
    The function:
    - Calculates where 'x' sits in the input range (as fraction 't')
    - Maps that fraction to the output range
    - Clamps to prevent out-of-bounds values
    """  # Docstring: explains the purpose and behavior of the scaler.
    if in_max <= in_min:                   # Guard: avoid divide-by-zero or invalid range.
        return out_min                     # If bad range, fall back to lowest output.
    t = (x - in_min) / (in_max - in_min)   # Normalize x into 0..1 fraction.
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)  # Clamp fraction to [0, 1].
    return int(round(out_min + t * (out_max - out_min)))  # Map to output range and quantize to int.


class BreathingRateMonitor:
    """Track breathing rate calculated from breath cycle durations AND from sensor channel 2."""
    def __init__(self):
        # Cycle-based calculations
        self.breath_durations = deque(maxlen=20)  # Store time between breaths
        self.cycle_br_readings = deque(maxlen=20) # Store calculated BrPM from cycles
        self.start_cycle_br = None
        self.ready_cycle = False
        self.breaths_detected = 0
        
        # Sensor channel 2 readings
        self.sensor_br_readings = deque(maxlen=20) # Store BrPM from sensor
        self.start_sensor_br = None
        self.ready_sensor = False
        self.sensor_reading_count = 0
        
        self.start_time = None
        
    def add_breath_duration(self, duration_seconds):
        """Add a new breath duration and calculate breathing rate from cycles."""
        # Validate duration is reasonable (0.5s to 30s per breath)
        if 0.5 <= duration_seconds <= 30.0:
            self.breath_durations.append(duration_seconds)
            self.breaths_detected += 1
            
            # Calculate BrPM from duration
            br = 60.0 / duration_seconds
            
            # Validate BrPM is in reasonable range
            if BR_VALID_RANGE[0] <= br <= BR_VALID_RANGE[1]:
                self.cycle_br_readings.append(br)
                
                # Mark as ready after enough breaths
                if not self.ready_cycle and self.breaths_detected >= BR_WARMUP_BREATHS:
                    self.ready_cycle = True
                    print(f"✓ Cycle-based breathing rate stabilized after {self.breaths_detected} breaths")
                
                # Set start BR once ready
                if self.start_cycle_br is None and self.ready_cycle:
                    self.start_cycle_br = self.get_current_cycle_br()
                    if self.start_time is None:
                        self.start_time = time.time()
                
                return br
        return None
    
    def add_sensor_reading(self, br_value):
        """Add a new breathing rate reading from sensor channel 2."""
        # Validate value is reasonable
        if br_value is not None and not np.isnan(br_value) and BR_VALID_RANGE[0] <= br_value <= BR_VALID_RANGE[1]:
            self.sensor_br_readings.append(br_value)
            self.sensor_reading_count += 1
            
            # Capture the FIRST valid reading as start value
            if self.start_sensor_br is None:
                self.start_sensor_br = br_value
                print(f"✓ Captured first sensor breathing rate: {br_value:.1f} BrPM")
                if self.start_time is None:
                    self.start_time = time.time()
            
            # Mark as ready after enough readings (but start is already captured)
            if not self.ready_sensor and self.sensor_reading_count >= 10:
                self.ready_sensor = True
                print(f"✓ Sensor-based breathing rate stabilized")
            
            return br_value
        return None
    
    def get_current_cycle_br(self):
        """Get smoothed current breathing rate from cycles using median of recent readings."""
        if len(self.cycle_br_readings) >= 3:
            return np.median(list(self.cycle_br_readings))
        return None
    
    def get_current_sensor_br(self):
        """Get smoothed current breathing rate from sensor using median of recent readings."""
        if len(self.sensor_br_readings) >= 3:
            return np.median(list(self.sensor_br_readings))
        return None
    
    def get_start_cycle_br(self):
        """Get the starting cycle-based breathing rate."""
        return self.start_cycle_br
    
    def get_start_sensor_br(self):
        """Get the starting sensor-based breathing rate."""
        return self.start_sensor_br
    
    def is_cycle_ready(self):
        """Check if we have stable cycle-based breathing rate data."""
        return self.ready_cycle
    
    def is_sensor_ready(self):
        """Check if we have stable sensor-based breathing rate data."""
        return self.ready_sensor

class BreathPerCycleNormalizer:
    """
    Detect troughs from EMA slope sign change (− → +) with a refractory period.
    On each detected trough, we finalize the previous cycle extremes (min/max)
    and maintain a rolling window of breath ranges for calibration.
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

        # Rolling window of breath ranges (min, max, time)
        self.breath_ranges = deque(maxlen=20)  # Store more than we need
        
        # Published range (averaged over window)
        self.in_min = None
        self.in_max = None

        # Breath timing tracking
        self.last_trough_time = None
        self.total_breaths = 0

    def sign(self, v):
        if v > self.deriv_eps: return 1
        if v < -self.deriv_eps: return -1
        return 0

    def update(self, x, t):
        """Update with a new EMA sample at time t (seconds). Returns (event, duration)."""
        # update running extrema
        self.cycle_min = min(self.cycle_min, x)
        self.cycle_max = max(self.cycle_max, x)

        if self.prev_x is None:
            self.prev_x = x
            return False, None  # no event yet

        d = x - self.prev_x
        s = self.sign(d)

        event = False
        duration = None
        
        if self.cool > 0:
            self.cool -= 1
        else:
            # detect trough: going down then up (− → +)
            if self.prev_sign < 0 and s > 0:
                span = self.cycle_max - self.cycle_min
                if span >= self.min_span:
                    # Store this breath's range
                    self.breath_ranges.append((self.cycle_min, self.cycle_max, t))
                    self.total_breaths += 1
                    
                    # Calculate averaged range over calibration window
                    window_size = global_params.get_calib_window_breaths()
                    recent_ranges = list(self.breath_ranges)[-window_size:]
                    
                    if recent_ranges:
                        mins = [r[0] for r in recent_ranges]
                        maxs = [r[1] for r in recent_ranges]
                        self.in_min = np.mean(mins)
                        self.in_max = np.mean(maxs)
                    
                    # Calculate breath duration
                    if self.last_trough_time is not None:
                        duration = t - self.last_trough_time
                    
                    self.last_trough_time = t
                    event = True
                    
                # reset for next cycle (start new tracking from current point)
                self.cycle_min = x
                self.cycle_max = x
                self.cool = self.refractory

        self.prev_sign = s
        self.prev_x = x
        return event, duration

    @property
    def min_span(self):
        return MIN_BREATH_SPAN

    def get_range(self):
        return self.in_min, self.in_max
    
    def get_breath_ranges(self):
        """Get the rolling window of breath ranges for visualization."""
        return list(self.breath_ranges)

class LivePlotter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.raw_vals = deque(maxlen=window_size)
        self.smooth_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)
        self.hr_vals = deque(maxlen=window_size)

        # Create figure with extra space for sliders
        self.fig = plt.figure(figsize=(12, 12))
        
        # Create grid for plots and sliders
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 2, 2, 2, 1], hspace=0.4, top=0.95, bottom=0.10)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax4 = self.fig.add_subplot(gs[3])
        
        self.fig.suptitle('Respiration Belt + Heart Rate → MIDI (Cycle-Based Breathing Rate)', fontsize=14, fontweight='bold')

        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        # Store calibration window patches
        self.calib_patches = []

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
        ax_sigmoid = plt.axes([0.12, 0.06, 0.25, 0.02])
        ax_alpha = plt.axes([0.12, 0.03, 0.25, 0.02])
        ax_calib = plt.axes([0.55, 0.045, 0.35, 0.02])
        
        self.slider_sigmoid = Slider(
            ax_sigmoid, 'Sigmoid\nStrength', 0.5, 10.0, 
            valinit=SIGMOID_STRENGTH, valstep=0.1
        )
        self.slider_alpha = Slider(
            ax_alpha, 'Smooth α', 0.01, 0.99, 
            valinit=SMOOTH_ALPHA, valstep=0.01
        )
        self.slider_calib = Slider(
            ax_calib, 'Calibration Window (breaths)', 1, 10,
            valinit=CALIB_WINDOW_BREATHS, valstep=1
        )
        
        # Connect slider events
        self.slider_sigmoid.on_changed(self.update_sigmoid)
        self.slider_alpha.on_changed(self.update_alpha)
        self.slider_calib.on_changed(self.update_calib_window)

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
    
    def update_calib_window(self, val):
        """Callback for calibration window slider."""
        global_params.set_calib_window_breaths(int(val))

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

    def update(self, info_text=None, breath_ranges=None):
        if not self.times:
            return
        times = list(self.times)
        self.line_raw.set_data(times, list(self.raw_vals))
        self.line_smooth.set_data(times, list(self.smooth_vals))
        self.ax1.relim(); self.ax1.autoscale_view()
        x_min = max(0, times[-1] - PLOT_WINDOW_SECS)
        x_max = times[-1] + 0.5
        self.ax1.set_xlim(x_min, x_max)
        
        # Update calibration window visualization
        for patch in self.calib_patches:
            patch.remove()
        self.calib_patches = []
        
        if breath_ranges:
            window_size = global_params.get_calib_window_breaths()
            recent_ranges = breath_ranges[-window_size:]
            
            # Get y-axis limits for shading
            y_min, y_max = self.ax1.get_ylim()
            
            # Draw vertical bands for each breath in the calibration window
            for i, (br_min, br_max, br_time) in enumerate(recent_ranges):
                # Estimate breath duration (use a default if this is the first breath)
                if i < len(recent_ranges) - 1:
                    breath_duration = recent_ranges[i + 1][2] - br_time
                else:
                    breath_duration = 3.0  # default ~3 seconds for the most recent breath
                
                # Only show if within plot window
                if br_time >= x_min and br_time <= x_max:
                    # Create a semi-transparent vertical band
                    alpha = 0.15 + (i / max(1, len(recent_ranges) - 1)) * 0.15  # Darker for more recent
                    rect = Rectangle((br_time, y_min), breath_duration, y_max - y_min,
                                   facecolor='yellow', alpha=alpha, edgecolor='orange', 
                                   linewidth=0.5, linestyle='--')
                    self.ax1.add_patch(rect)
                    self.calib_patches.append(rect)

        self.line_midi.set_data(times, list(self.midi_vals))
        self.ax2.set_xlim(x_min, x_max)

        if self.hr_vals:
            hr_times = times[-len(self.hr_vals):]
            self.line_hr.set_data(hr_times, list(self.hr_vals))
            self.ax3.relim(); self.ax3.autoscale_view(scaley=True, scalex=False)
            self.ax3.set_xlim(x_min, x_max)

        # Update sigmoid curve
        self.update_sigmoid_curve()

        if info_text:
            # Add slider values to info text
            strength = global_params.get_sigmoid_strength()
            alpha = global_params.get_smooth_alpha()
            calib_window = global_params.get_calib_window_breaths()
            info_text += f"\nSigmoid: {strength:.1f} | Alpha: {alpha:.2f} | Calib: {calib_window} breaths"
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
    
    # Select sensors - try both, gracefully handle if sensor 2 fails
    try:
        print(f"Selecting sensors: {SENSOR_CHANNELS}")
        g.select_sensors(SENSOR_CHANNELS)
        print(f"✓ Successfully selected sensors: {SENSOR_CHANNELS}")
    except Exception as e:
        print(f"Warning: Could not select all sensors: {e}")
        print("Trying sensor 1 (Force) only...")
        try:
            g.select_sensors([1])
            print("✓ Selected sensor 1 (Force) - sensor 2 breathing rate unavailable")
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
    
    # Setup breathing rate monitor (cycle-based only)
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
        
        raw = data[0]  # Force sensor
        
        # Try to get breathing rate from sensor 2 if available
        if len(data) > 1:
            sensor_br = data[1]
            if sensor_br is not None:
                br_monitor.add_sensor_reading(sensor_br)
        
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
    start_sensor_br = br_monitor.get_start_sensor_br()
    
    print("\n" + "="*60)
    print("SESSION START METRICS")
    print("="*60)
    if start_hr:
        print(f"Heart Rate:              {start_hr:.1f} BPM")
    else:
        print(f"Heart Rate:              Detecting...")
    print(f"Breathing Rate (Cycles): Waiting for {BR_WARMUP_BREATHS} breaths...")
    if start_sensor_br:
        print(f"Breathing Rate (Sensor): {start_sensor_br:.1f} BrPM")
    else:
        print(f"Breathing Rate (Sensor): Detecting...")
    print(f"Calibration Window:      {global_params.get_calib_window_breaths()} breaths")
    print("="*60 + "\n")

    # current range
    in_min, in_max = None, None
    last_sent = None
    frame = 0

    print("Streaming. Use sliders to adjust parameters. Press Ctrl+C to stop.\n")
    print("Note: Comparing cycle-based BR (calculated) vs sensor-based BR (channel 2).\n")
    
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]  # Force sensor
            
            # Get breathing rate from sensor 2 if available
            sensor_br = None
            if len(data) > 1:
                sensor_br_raw = data[1]
                if sensor_br_raw is not None:
                    sensor_br = br_monitor.add_sensor_reading(sensor_br_raw)
            
            # Use current smooth alpha from global params
            alpha = global_params.get_smooth_alpha()
            ema = raw if ema is None else (alpha * ema + (1 - alpha) * raw)

            # Update heart rate
            current_hr = hr_monitor.update()
            
            # Update per-breath detector and get breath duration
            event, breath_duration = norm.update(ema, time.time())
            
            # If breath detected, update breathing rate from cycle duration
            cycle_br = None
            if event and breath_duration is not None:
                cycle_br = br_monitor.add_breath_duration(breath_duration)
                in_min, in_max = norm.get_range()
                
                if in_min is not None and in_max is not None:
                    cycle_br_display = br_monitor.get_current_cycle_br()
                    sensor_br_display = br_monitor.get_current_sensor_br()
                    hr_display = hr_monitor.get_current_hr()
                    calib_window = global_params.get_calib_window_breaths()
                    duration_display = breath_duration
                    
                    # Build output string
                    output = f"Breath #{norm.total_breaths} → Duration: {duration_display:.2f}s | "
                    output += f"Range: [{in_min:.5f}, {in_max:.5f}] (avg {calib_window})\n"
                    output += f"  BR (Cycles): {(cycle_br_display or 0):.1f} BrPM"
                    if sensor_br_display:
                        output += f" | BR (Sensor): {sensor_br_display:.1f} BrPM"
                    else:
                        output += f" | BR (Sensor): N/A"
                    output += f" | HR: {(hr_display or 0):.1f} BPM"
                    print(output)

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
                cycle_br_display = br_monitor.get_current_cycle_br()
                sensor_br_display = br_monitor.get_current_sensor_br()
                
                info = ""
                # Cycle-based BR
                if br_monitor.is_cycle_ready() and cycle_br_display:
                    info = f"BR (Cycles): {cycle_br_display:.1f} BrPM"
                else:
                    breaths_needed = BR_WARMUP_BREATHS - br_monitor.breaths_detected
                    if breaths_needed > 0:
                        info = f"BR (Cycles): waiting ({breaths_needed} more breaths)"
                    else:
                        info = "BR (Cycles): stabilizing..."
                
                # Sensor-based BR
                if br_monitor.is_sensor_ready() and sensor_br_display:
                    info += f"\nBR (Sensor): {sensor_br_display:.1f} BrPM"
                else:
                    info += f"\nBR (Sensor): stabilizing..."
                
                info += f"\nTotal breaths: {norm.total_breaths}"
                info += f"\nRange: [{in_min:.3f}, {in_max:.3f}]"
                
                if current_hr:
                    info += f"\nHeart Rate: {current_hr:.1f} BPM"
                
                # Get breath ranges for visualization
                breath_ranges = norm.get_breath_ranges()
                plotter.update(info, breath_ranges)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\n\nStopping…")
    finally:
        # Display end metrics
        end_cycle_br = br_monitor.get_current_cycle_br()
        end_sensor_br = br_monitor.get_current_sensor_br()
        end_hr = hr_monitor.get_current_hr()
        
        print("\n" + "="*60)
        print("SESSION END METRICS")
        print("="*60)
        if end_hr:
            print(f"Heart Rate:              {end_hr:.1f} BPM")
        else:
            print(f"Heart Rate:              Not available")
        if end_cycle_br:
            print(f"Breathing Rate (Cycles): {end_cycle_br:.1f} BrPM")
        else:
            print(f"Breathing Rate (Cycles): Not available")
        if end_sensor_br:
            print(f"Breathing Rate (Sensor): {end_sensor_br:.1f} BrPM")
        else:
            print(f"Breathing Rate (Sensor): Not available")
        print(f"Total Breaths:           {norm.total_breaths}")
        print("="*60)
        
        # Get start metrics
        start_cycle_br = br_monitor.get_start_cycle_br()
        start_sensor_br = br_monitor.get_start_sensor_br()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print("START:")
        if start_hr:
            print(f"  Heart Rate:              {start_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:              Not available")
        if start_cycle_br:
            print(f"  Breathing Rate (Cycles): {start_cycle_br:.1f} BrPM")
        else:
            print(f"  Breathing Rate (Cycles): Not available (needed {BR_WARMUP_BREATHS} breaths)")
        if start_sensor_br:
            print(f"  Breathing Rate (Sensor): {start_sensor_br:.1f} BrPM")
        else:
            print(f"  Breathing Rate (Sensor): Not available")
        
        print("\nEND:")
        if end_hr:
            print(f"  Heart Rate:              {end_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:              Not available")
        if end_cycle_br:
            print(f"  Breathing Rate (Cycles): {end_cycle_br:.1f} BrPM")
        else:
            print(f"  Breathing Rate (Cycles): Not available")
        if end_sensor_br:
            print(f"  Breathing Rate (Sensor): {end_sensor_br:.1f} BrPM")
        else:
            print(f"  Breathing Rate (Sensor): Not available")
        
        print("\nCHANGES:")
        if start_hr and end_hr:
            hr_change = end_hr - start_hr
            print(f"  Heart Rate Change:              {hr_change:+.1f} BPM")
        if start_cycle_br and end_cycle_br:
            cycle_br_change = end_cycle_br - start_cycle_br
            print(f"  Breathing Rate Change (Cycles): {cycle_br_change:+.1f} BrPM")
        elif not start_cycle_br:
            print(f"  Breathing Rate Change (Cycles): Not available (needed {BR_WARMUP_BREATHS} breaths at start)")
        if start_sensor_br and end_sensor_br:
            sensor_br_change = end_sensor_br - start_sensor_br
            print(f"  Breathing Rate Change (Sensor): {sensor_br_change:+.1f} BrPM")
        elif not start_sensor_br:
            print(f"  Breathing Rate Change (Sensor): Not available (no valid sensor readings)")
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