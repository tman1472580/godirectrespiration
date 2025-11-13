#!/usr/bin/env python3
"""
Go Direct Spirometer (Flow Rate) → MIDI CC (0–127 full-span per breath) + Polar H10 Heart Rate Monitor

- Creates a virtual MIDI port. Select it in your DAW and map to CC7 (Volume) or any CC.
- Uses flow rate sensor (L/s) and integrates to get breath volume.
- Detects breath boundaries using zero-crossings of flow rate.
- Calculates breathing rate from detected breath cycle durations.
- Monitors heart rate via Polar H10 Bluetooth device.
- Shows start and end metrics for both heart rate and breathing rate.
- Shows live plot of flow rate, integrated volume, MIDI CC, and heart rate.
- Interactive sliders for sigmoid strength, smoothing alpha, and calibration window.

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
VIRTUAL_PORT_NAME = "Spirometer CC (Python)"
MIDI_CHANNEL       = 1          # 1–16
MIDI_CC            = 7          # CC number to send
READ_HZ            = 100        # sensor sample rate (Hz)
SMOOTH_ALPHA       = 0.3        # EMA: ema = α*ema + (1-α)*raw (α→1 = heavier smoothing)
MIN_SEND_STEP      = 0          # only send CC if change ≥ this amount
INITIAL_CALIB_SECS = 10         # quick warm-up to seed EMA and plot
SENSOR_CHANNELS    = [1, 9]     # Flow rate + respiration rate sensor
PLOT_WINDOW_SECS   = 10

# Breath detection settings for flow rate
ZERO_CROSSING_THRESHOLD = 0.05  # L/s - threshold for detecting zero crossing
REFRACTORY_MS      = 250        # ignore new breaths for this time after detection
MIN_BREATH_VOLUME  = 0.1        # minimum volume (L) to accept as a "breath"
BPM_ROLLING        = 6          # average last N breath periods for BPM

# Polar H10 settings
POLAR_DEVICE_NAME  = "Polar H10"  # Device name prefix to search for
HR_WARMUP_SECS     = 5            # Time to collect initial HR readings

# Breathing rate settings
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
    """Linear scaling from input range to output range with clamping."""
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return int(round(out_min + t * (out_max - out_min)))

class BreathingRateMonitor:
    """Track breathing rate calculated from breath cycle durations AND from sensor channel."""
    def __init__(self):
        # Cycle-based calculations
        self.breath_durations = deque(maxlen=20)
        self.cycle_br_readings = deque(maxlen=20)
        self.start_cycle_br = None
        self.ready_cycle = False
        self.breaths_detected = 0
        
        # Sensor channel readings
        self.sensor_br_readings = deque(maxlen=20)
        self.start_sensor_br = None
        self.ready_sensor = False
        self.sensor_reading_count = 0
        
        self.start_time = None
        
    def add_breath_duration(self, duration_seconds):
        """Add a new breath duration and calculate breathing rate from cycles."""
        if 0.5 <= duration_seconds <= 30.0:
            self.breath_durations.append(duration_seconds)
            self.breaths_detected += 1
            
            br = 60.0 / duration_seconds
            
            if BR_VALID_RANGE[0] <= br <= BR_VALID_RANGE[1]:
                self.cycle_br_readings.append(br)
                
                if not self.ready_cycle and self.breaths_detected >= BR_WARMUP_BREATHS:
                    self.ready_cycle = True
                    print(f"✓ Cycle-based breathing rate stabilized after {self.breaths_detected} breaths")
                
                if self.start_cycle_br is None and self.ready_cycle:
                    self.start_cycle_br = self.get_current_cycle_br()
                    if self.start_time is None:
                        self.start_time = time.time()
                
                return br
        return None
    
    def add_sensor_reading(self, br_value):
        """Add a new breathing rate reading from sensor channel."""
        if br_value is not None and not np.isnan(br_value) and BR_VALID_RANGE[0] <= br_value <= BR_VALID_RANGE[1]:
            self.sensor_br_readings.append(br_value)
            self.sensor_reading_count += 1
            
            if self.start_sensor_br is None:
                self.start_sensor_br = br_value
                print(f"✓ Captured first sensor breathing rate: {br_value:.1f} BrPM")
                if self.start_time is None:
                    self.start_time = time.time()
            
            if not self.ready_sensor and self.sensor_reading_count >= 10:
                self.ready_sensor = True
                print(f"✓ Sensor-based breathing rate stabilized")
            
            return br_value
        return None
    
    def get_current_cycle_br(self):
        """Get smoothed current breathing rate from cycles."""
        if len(self.cycle_br_readings) >= 3:
            return np.median(list(self.cycle_br_readings))
        return None
    
    def get_current_sensor_br(self):
        """Get smoothed current breathing rate from sensor."""
        if len(self.sensor_br_readings) >= 3:
            return np.median(list(self.sensor_br_readings))
        return None
    
    def get_start_cycle_br(self):
        return self.start_cycle_br
    
    def get_start_sensor_br(self):
        return self.start_sensor_br
    
    def is_cycle_ready(self):
        return self.ready_cycle
    
    def is_sensor_ready(self):
        return self.ready_sensor

class FlowRateBreathDetector:
    """
    Detect breaths from flow rate using zero-crossings and volume integration.
    Flow rate is positive during inhalation, negative during exhalation.
    """
    def __init__(self, fs, zero_threshold=0.05, refractory_ms=250, min_volume=0.1):
        self.fs = fs
        self.dt = 1.0 / fs  # time step in seconds
        self.zero_threshold = zero_threshold
        self.refractory = int(round(refractory_ms * fs / 1000.0))
        self.cool = 0
        self.min_volume = min_volume
        
        # State tracking
        self.prev_flow = 0
        self.was_positive = False
        self.current_volume = 0.0  # integrated volume for current breath
        
        # Breath tracking
        self.breath_volumes = deque(maxlen=20)
        self.breath_ranges = deque(maxlen=20)  # (min_vol, max_vol, time)
        self.last_breath_time = None
        self.total_breaths = 0
        
        # Calibration ranges
        self.in_min = None
        self.in_max = None
    
    def update(self, flow, t):
        """
        Update with flow rate (L/s) at time t (seconds).
        Returns (breath_detected, breath_duration, breath_volume).
        """
        # Integrate flow to get volume change
        self.current_volume += flow * self.dt
        
        event = False
        duration = None
        volume = None
        
        if self.cool > 0:
            self.cool -= 1
        else:
            # Detect zero crossing from negative to positive (end of exhalation)
            is_positive = flow > self.zero_threshold
            
            if self.was_positive is not None and not self.was_positive and is_positive:
                # Zero crossing detected - end of breath cycle
                breath_vol = abs(self.current_volume)
                
                if breath_vol >= self.min_volume:
                    # Valid breath detected
                    self.breath_volumes.append(breath_vol)
                    self.total_breaths += 1
                    volume = breath_vol
                    
                    # Store breath range
                    self.breath_ranges.append((0, breath_vol, t))
                    
                    # Calculate averaged range over calibration window
                    window_size = global_params.get_calib_window_breaths()
                    recent_ranges = list(self.breath_ranges)[-window_size:]
                    
                    if recent_ranges:
                        volumes = [r[1] for r in recent_ranges]
                        self.in_min = 0  # Flow volume starts at 0
                        self.in_max = np.mean(volumes)
                    
                    # Calculate breath duration
                    if self.last_breath_time is not None:
                        duration = t - self.last_breath_time
                    
                    self.last_breath_time = t
                    event = True
                    self.cool = self.refractory
                
                # Reset volume integration for next breath
                self.current_volume = 0.0
            
            self.was_positive = is_positive
        
        self.prev_flow = flow
        return event, duration, volume
    
    def get_range(self):
        """Get current calibration range for volume."""
        return self.in_min, self.in_max
    
    def get_breath_ranges(self):
        """Get the rolling window of breath ranges for visualization."""
        return list(self.breath_ranges)
    
    def get_current_volume(self):
        """Get current integrated volume within breath."""
        return abs(self.current_volume)

class LivePlotter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.flow_vals = deque(maxlen=window_size)
        self.volume_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)
        self.hr_vals = deque(maxlen=window_size)

        # Create figure with extra space for sliders
        self.fig = plt.figure(figsize=(12, 12))
        
        gs = self.fig.add_gridspec(5, 1, height_ratios=[3, 2, 2, 2, 1], hspace=0.4, top=0.95, bottom=0.10)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax4 = self.fig.add_subplot(gs[3])
        
        self.fig.suptitle('Spirometer Flow Rate + Heart Rate → MIDI', fontsize=14, fontweight='bold')

        self.line_flow, = self.ax1.plot([], [], 'b-', linewidth=2, label='Flow Rate (L/s)')
        self.line_volume, = self.ax1.plot([], [], 'c-', alpha=0.5, label='Integrated Volume (L)')
        self.ax1.set_ylabel('Flow Rate (L/s) / Volume (L)')
        self.ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        # Store calibration window patches
        self.calib_patches = []

        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC')
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
        
        self.slider_sigmoid.on_changed(self.update_sigmoid)
        self.slider_alpha.on_changed(self.update_alpha)
        self.slider_calib.on_changed(self.update_calib_window)

        self.start_time = time.time()
        self.update_sigmoid_curve()
        plt.ion(); plt.show()

    def update_sigmoid(self, val):
        global_params.set_sigmoid_strength(val)
        self.update_sigmoid_curve()

    def update_alpha(self, val):
        global_params.set_smooth_alpha(val)
    
    def update_calib_window(self, val):
        global_params.set_calib_window_breaths(int(val))

    def update_sigmoid_curve(self):
        strength = global_params.get_sigmoid_strength()
        sigmoid_y = 1 / (1 + np.exp(-strength * (self.sigmoid_x - 0.5)))
        self.line_sigmoid.set_data(self.sigmoid_x, sigmoid_y)

    def add(self, flow, volume, midi, hr=None):
        t = time.time() - self.start_time
        self.times.append(t)
        self.flow_vals.append(flow)
        self.volume_vals.append(volume)
        self.midi_vals.append(midi)
        if hr is not None:
            self.hr_vals.append(hr)

    def update(self, info_text=None, breath_ranges=None):
        if not self.times:
            return
        times = list(self.times)
        self.line_flow.set_data(times, list(self.flow_vals))
        self.line_volume.set_data(times, list(self.volume_vals))
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
            
            y_min, y_max = self.ax1.get_ylim()
            
            for i, (br_min, br_max, br_time) in enumerate(recent_ranges):
                if i < len(recent_ranges) - 1:
                    breath_duration = recent_ranges[i + 1][2] - br_time
                else:
                    breath_duration = 3.0
                
                if br_time >= x_min and br_time <= x_max:
                    alpha = 0.15 + (i / max(1, len(recent_ranges) - 1)) * 0.15
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

        self.update_sigmoid_curve()

        if info_text:
            strength = global_params.get_sigmoid_strength()
            alpha = global_params.get_smooth_alpha()
            calib_window = global_params.get_calib_window_breaths()
            info_text += f"\nSigmoid: {strength:.1f} | Alpha: {alpha:.2f} | Calib: {calib_window} breaths"
            self.info_text.set_text(info_text)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def connect_device():
    """Connect to Go Direct Spirometer synchronously."""
    g = gdx.gdx()
    print("\nConnecting to Go Direct Spirometer...")
    
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
    
    # Select sensors
    try:
        print(f"Selecting sensors: {SENSOR_CHANNELS}")
        g.select_sensors(SENSOR_CHANNELS)
        print(f"✓ Successfully selected sensors: {SENSOR_CHANNELS}")
    except Exception as e:
        print(f"Warning: Could not select all sensors: {e}")
        print("Trying sensor 1 (Flow Rate) only...")
        try:
            g.select_sensors([1])
            print("✓ Selected sensor 1 (Flow Rate) - sensor 9 respiration rate unavailable")
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
            recent_ibis = self.polar.ibi_stream_values[-10:]
            if recent_ibis:
                avg_ibi_ms = np.mean(recent_ibis)
                hr = 60000.0 / avg_ibi_ms
                self.hr_readings.append(hr)
                
                if self.start_hr is None and len(self.hr_readings) >= 5:
                    self.start_hr = np.mean(list(self.hr_readings))
                    self.start_time = time.time()
                    
                return hr
        return None
    
    def get_current_hr(self):
        with self.lock:
            if self.hr_readings:
                return np.mean(list(self.hr_readings))
        return None
    
    def get_start_hr(self):
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
    # Connect to spirometer
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

    # --- quick warmup to collect initial data ---
    print(f"\nWarm-up {INITIAL_CALIB_SECS}s...")
    t0 = time.time()
    flow_ema = None
    warm_data = []
    
    while time.time() - t0 < INITIAL_CALIB_SECS:
        data = g.read()
        if not data:
            time.sleep(1.0 / READ_HZ)
            continue
        
        flow_raw = data[0]  # Flow rate sensor
        
        # Try to get breathing rate from sensor 9 if available
        if len(data) > 1:
            sensor_br = data[1]
            if sensor_br is not None:
                br_monitor.add_sensor_reading(sensor_br)
        
        # Smooth flow rate
        alpha = global_params.get_smooth_alpha()
        flow_ema = flow_raw if flow_ema is None else (alpha * flow_ema + (1 - alpha) * flow_raw)
        warm_data.append((flow_raw, flow_ema))
        hr_monitor.update()
        time.sleep(1.0 / READ_HZ)
    
    print("Warm-up done.")

    # --- plotter & detector ---
    plotter = LivePlotter(window_size=int(PLOT_WINDOW_SECS * READ_HZ))
    detector = FlowRateBreathDetector(
        fs=READ_HZ,
        zero_threshold=ZERO_CROSSING_THRESHOLD,
        refractory_ms=REFRACTORY_MS,
        min_volume=MIN_BREATH_VOLUME
    )

    # seed plot with warm data
    for flow_raw, flow_ema in warm_data[-int(READ_HZ):]:
        detector.update(flow_ema, time.time())
        plotter.add(flow_raw, 0, 0, hr_monitor.get_current_hr())

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
    
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            flow_raw = data[0]  # Flow rate sensor
            
            # Get breathing rate from sensor 9 if available
            sensor_br = None
            if len(data) > 1:
                sensor_br_raw = data[1]
                if sensor_br_raw is not None:
                    sensor_br = br_monitor.add_sensor_reading(sensor_br_raw)
            
            # Smooth flow rate
            alpha = global_params.get_smooth_alpha()
            flow_ema = flow_raw if flow_ema is None else (alpha * flow_ema + (1 - alpha) * flow_raw)

            # Update heart rate
            current_hr = hr_monitor.update()
            
            # Update breath detector and get breath event
            event, breath_duration, breath_volume = detector.update(flow_ema, time.time())
            
            # If breath detected, update breathing rate from cycle duration
            cycle_br = None
            if event and breath_duration is not None:
                cycle_br = br_monitor.add_breath_duration(breath_duration)
                in_min, in_max = detector.get_range()
                
                if in_min is not None and in_max is not None:
                    cycle_br_display = br_monitor.get_current_cycle_br()
                    sensor_br_display = br_monitor.get_current_sensor_br()
                    hr_display = hr_monitor.get_current_hr()
                    calib_window = global_params.get_calib_window_breaths()
                    
                    # Build output string
                    output = f"Breath #{detector.total_breaths} → Duration: {breath_duration:.2f}s | Volume: {breath_volume:.3f}L\n"
                    output += f"  Range: [0, {in_max:.3f}L] (avg {calib_window} breaths)\n"
                    output += f"  BR (Cycles): {(cycle_br_display or 0):.1f} BrPM"
                    if sensor_br_display:
                        output += f" | BR (Sensor): {sensor_br_display:.1f} BrPM"
                    else:
                        output += f" | BR (Sensor): N/A"
                    output += f" | HR: {(hr_display or 0):.1f} BPM"
                    print(output)

            # Get current integrated volume for MIDI mapping
            current_volume = detector.get_current_volume()
            
            # If we still don't have a range (first cycle), use a default
            if in_min is None or in_max is None or in_max <= in_min:
                in_min = 0
                in_max = 1.0  # 1 liter default

            # Map volume → 0..127 MIDI CC
            cc_val = linear_scale(current_volume, in_min, in_max, 0, 127)

            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                _midi_out.send(mido.Message('control_change',
                                            channel=MIDI_CHANNEL - 1,
                                            control=MIDI_CC,
                                            value=cc_val))
                last_sent = cc_val

            # Plot
            plotter.add(flow_raw, current_volume, cc_val, current_hr)
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
                
                info += f"\nTotal breaths: {detector.total_breaths}"
                info += f"\nCurrent volume: {current_volume:.3f}L"
                if in_max:
                    info += f"\nRange: [0, {in_max:.3f}L]"
                
                if current_hr:
                    info += f"\nHeart Rate: {current_hr:.1f} BPM"
                
                # Get breath ranges for visualization
                breath_ranges = detector.get_breath_ranges()
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
        print(f"Total Breaths:           {detector.total_breaths}")
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