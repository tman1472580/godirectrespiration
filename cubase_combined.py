#!/usr/bin/env python3
"""
Go Direct Respiration Belt → MIDI CC with Dual Calibration Modes + Polar H10 Heart Rate

Features:
- Two calibration modes: Fixed Window (10s) or Dynamic Breath-Based
- Creates a virtual MIDI port for DAW integration
- Monitors heart rate via Polar H10 Bluetooth
- Live plotting of signals with interactive controls
- Toggle between calibration modes during runtime

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
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from bleak import BleakScanner
from PolarH10 import PolarH10

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"
MIDI_CHANNEL       = 1
MIDI_CC            = 7
READ_HZ            = 100
SMOOTH_ALPHA       = 0.3
MIN_SEND_STEP      = 0
INITIAL_CALIB_SECS = 10
SENSOR_CHANNELS    = [1, 2]
PLOT_WINDOW_SECS   = 10

# Peak/trough detection
DERIV_EPS          = 1e-5
REFRACTORY_MS      = 250
MIN_BREATH_SPAN    = 1e-3
BPM_ROLLING        = 6

# Polar H10 settings
POLAR_DEVICE_NAME  = "Polar H10"
HR_WARMUP_SECS     = 5

# Breathing rate settings
BR_WARMUP_BREATHS  = 5
BR_VALID_RANGE     = (2.0, 60.0)

# Calibration settings
FIXED_CALIB_SECS   = 15
CALIB_WINDOW_BREATHS = 3

# ==========================

mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")

class GlobalParams:
    def __init__(self):
        self.smooth_alpha = SMOOTH_ALPHA
        self.calib_window_breaths = CALIB_WINDOW_BREATHS
        self.calib_mode = "dynamic"  # "fixed" or "dynamic"
        self.recalibrate_flag = False
        self.lock = threading.Lock()
    
    def set_smooth_alpha(self, value):
        with self.lock:
            self.smooth_alpha = value
    
    def get_smooth_alpha(self):
        with self.lock:
            return self.smooth_alpha
    
    def set_calib_window_breaths(self, value):
        with self.lock:
            self.calib_window_breaths = int(value)
    
    def get_calib_window_breaths(self):
        with self.lock:
            return self.calib_window_breaths
    
    def set_calib_mode(self, mode):
        with self.lock:
            self.calib_mode = mode
    
    def get_calib_mode(self):
        with self.lock:
            return self.calib_mode
    
    def request_recalibrate(self):
        with self.lock:
            self.recalibrate_flag = True
    
    def clear_recalibrate_flag(self):
        with self.lock:
            self.recalibrate_flag = False
    
    def should_recalibrate(self):
        with self.lock:
            return self.recalibrate_flag

global_params = GlobalParams()

def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    """Linear scaling from input range to MIDI CC range (0-127)."""
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return int(round(out_min + t * (out_max - out_min)))

class BreathingRateMonitor:
    """Track breathing rate from breath cycles and sensor."""
    def __init__(self):
        self.breath_durations = deque(maxlen=20)
        self.cycle_br_readings = deque(maxlen=20)
        self.start_cycle_br = None
        self.ready_cycle = False
        self.breaths_detected = 0
        
        self.sensor_br_readings = deque(maxlen=20)
        self.start_sensor_br = None
        self.ready_sensor = False
        self.sensor_reading_count = 0
        
        self.start_time = None
        
    def add_breath_duration(self, duration_seconds):
        """Calculate breathing rate from breath cycle duration."""
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
        """Add breathing rate reading from sensor channel 2."""
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
        if len(self.cycle_br_readings) >= 3:
            return np.median(list(self.cycle_br_readings))
        return None
    
    def get_current_sensor_br(self):
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

class BreathPerCycleNormalizer:
    """Detect breath troughs and maintain calibration ranges."""
    def __init__(self, fs, deriv_eps=1e-5, refractory_ms=250, min_span=1e-3, bpm_window=6):
        self.fs = fs
        self.deriv_eps = deriv_eps
        self.refractory = int(round(refractory_ms * fs / 1000.0))
        self.cool = 0

        self.prev_x = None
        self.prev_sign = 0

        self.cycle_min = float('inf')
        self.cycle_max = float('-inf')

        self.breath_ranges = deque(maxlen=20)
        
        self.in_min = None
        self.in_max = None

        self.last_trough_time = None
        self.total_breaths = 0

    def sign(self, v):
        if v > self.deriv_eps: return 1
        if v < -self.deriv_eps: return -1
        return 0

    def update(self, x, t):
        """Update with new EMA sample. Returns (event, duration)."""
        self.cycle_min = min(self.cycle_min, x)
        self.cycle_max = max(self.cycle_max, x)

        if self.prev_x is None:
            self.prev_x = x
            return False, None

        d = x - self.prev_x
        s = self.sign(d)

        event = False
        duration = None
        
        if self.cool > 0:
            self.cool -= 1
        else:
            if self.prev_sign < 0 and s > 0:
                span = self.cycle_max - self.cycle_min
                if span >= MIN_BREATH_SPAN:
                    self.breath_ranges.append((self.cycle_min, self.cycle_max, t))
                    self.total_breaths += 1
                    
                    # Update range based on calibration mode
                    mode = global_params.get_calib_mode()
                    if mode == "dynamic":
                        window_size = global_params.get_calib_window_breaths()
                        recent_ranges = list(self.breath_ranges)[-window_size:]
                    else:  # fixed mode - use all collected ranges
                        recent_ranges = list(self.breath_ranges)
                    
                    if recent_ranges:
                        mins = [r[0] for r in recent_ranges]
                        maxs = [r[1] for r in recent_ranges]
                        self.in_min = np.mean(mins)
                        self.in_max = np.mean(maxs)
                    
                    if self.last_trough_time is not None:
                        duration = t - self.last_trough_time
                    
                    self.last_trough_time = t
                    event = True
                    
                self.cycle_min = x
                self.cycle_max = x
                self.cool = self.refractory

        self.prev_sign = s
        self.prev_x = x
        return event, duration

    def get_range(self):
        return self.in_min, self.in_max
    
    def get_breath_ranges(self):
        return list(self.breath_ranges)
    
    def set_fixed_range(self, in_min, in_max):
        """Manually set calibration range (for fixed mode)."""
        self.in_min = in_min
        self.in_max = in_max

class LivePlotter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.raw_vals = deque(maxlen=window_size)
        self.smooth_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)
        self.hr_vals = deque(maxlen=window_size)

        self.fig = plt.figure(figsize=(12, 10))
        
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 2, 2, 1], hspace=0.4, top=0.95, bottom=0.12)
        
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        
        self.fig.suptitle('Respiration Belt + Heart Rate → MIDI', fontsize=14, fontweight='bold')

        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
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

        self.info_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # Create controls
        ax_alpha = plt.axes([0.12, 0.07, 0.25, 0.02])
        ax_calib = plt.axes([0.12, 0.04, 0.25, 0.02])
        ax_mode_btn = plt.axes([0.50, 0.055, 0.15, 0.04])
        ax_recal_btn = plt.axes([0.70, 0.055, 0.18, 0.04])
        
        self.slider_alpha = Slider(
            ax_alpha, 'Smooth α', 0.01, 0.99, 
            valinit=SMOOTH_ALPHA, valstep=0.01
        )
        self.slider_calib = Slider(
            ax_calib, 'Calib Window (breaths)', 1, 10,
            valinit=CALIB_WINDOW_BREATHS, valstep=1
        )
        
        self.btn_mode = Button(ax_mode_btn, 'Mode: Dynamic')
        self.btn_recalibrate = Button(ax_recal_btn, 'Recalibrate (10s)')
        
        self.slider_alpha.on_changed(self.update_alpha)
        self.slider_calib.on_changed(self.update_calib_window)
        self.btn_mode.on_clicked(self.toggle_mode)
        self.btn_recalibrate.on_clicked(self.trigger_recalibrate)

        self.start_time = time.time()
        plt.ion(); plt.show()

    def update_alpha(self, val):
        global_params.set_smooth_alpha(val)
    
    def update_calib_window(self, val):
        global_params.set_calib_window_breaths(int(val))

    def toggle_mode(self, event):
        mode = global_params.get_calib_mode()
        new_mode = "fixed" if mode == "dynamic" else "dynamic"
        global_params.set_calib_mode(new_mode)
        self.btn_mode.label.set_text(f"Mode: {new_mode.capitalize()}")
        print(f"\n→ Switched to {new_mode.upper()} calibration mode")
        
        # Update slider state
        if new_mode == "fixed":
            self.slider_calib.set_active(False)
        else:
            self.slider_calib.set_active(True)
    
    def trigger_recalibrate(self, event):
        if global_params.get_calib_mode() == "fixed":
            global_params.request_recalibrate()
            print(f"\n→ Recalibration requested ({FIXED_CALIB_SECS}s window)...")

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
        
        # Update calibration window visualization (only for dynamic mode)
        for patch in self.calib_patches:
            patch.remove()
        self.calib_patches = []
        
        mode = global_params.get_calib_mode()
        if breath_ranges and mode == "dynamic":
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

        if info_text:
            alpha = global_params.get_smooth_alpha()
            mode = global_params.get_calib_mode()
            info_text += f"\nMode: {mode.upper()} | Alpha: {alpha:.2f}"
            if mode == "dynamic":
                calib_window = global_params.get_calib_window_breaths()
                info_text += f" | Calib: {calib_window} breaths"
            self.info_text.set_text(info_text)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def connect_device():
    """Connect to Go Direct device."""
    g = gdx.gdx()
    print("\nConnecting to Go Direct Respiration Belt...")
    
    try:
        g.open(connection='ble')
        time.sleep(0.8)
    except Exception as e:
        print(f"BLE connection failed: {e}")
    
    if not getattr(g, "devices", None):
        try:
            print("Trying USB connection...")
            g.open(connection='usb')
            time.sleep(0.5)
        except Exception as e:
            print(f"USB connection failed: {e}")
            return None
    
    try:
        print(f"Selecting sensors: {SENSOR_CHANNELS}")
        g.select_sensors(SENSOR_CHANNELS)
        print(f"✓ Successfully selected sensors: {SENSOR_CHANNELS}")
    except Exception as e:
        print(f"Warning: Could not select all sensors: {e}")
        try:
            g.select_sensors([1])
            print("✓ Selected sensor 1 (Force) only")
        except Exception as e2:
            print(f"Failed to select any sensors: {e2}")
            return None
    
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
    """Start Polar H10 in separate thread."""
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

def do_calibration(g, duration_secs):
    """Perform fixed-window calibration and return min/max range."""
    print(f"Calibrating for {duration_secs}s... breathe normally, then take deep breaths.")
    t0 = time.time()
    calib_vals = []
    
    while time.time() - t0 < duration_secs:
        data = g.read()
        if data:
            calib_vals.append(data[0])
        time.sleep(1.0 / READ_HZ)
    
    if not calib_vals:
        return None, None
    
    raw_min = min(calib_vals)
    raw_max = max(calib_vals)
    pad = 0.05 * (raw_max - raw_min or 1.0)
    in_min = raw_min - pad
    in_max = raw_max + pad
    
    print(f"Calibration complete. Range: [{in_min:.3f}, {in_max:.3f}]")
    return in_min, in_max

def main():
    g = connect_device()
    if g is None:
        print("❌ No device. Exiting.")
        return

    hr_monitor = HeartRateMonitor()
    polar_thread = start_polar_thread(hr_monitor)
    br_monitor = BreathingRateMonitor()
    
    time.sleep(HR_WARMUP_SECS + 2)

    # Initial warmup
    print(f"\nWarm-up {INITIAL_CALIB_SECS}s...")
    t0 = time.time()
    ema = None
    warm_raw = []
    
    while time.time() - t0 < INITIAL_CALIB_SECS:
        data = g.read()
        if not data:
            time.sleep(1.0 / READ_HZ)
            continue
        
        raw = data[0]
        
        if len(data) > 1:
            sensor_br = data[1]
            if sensor_br is not None:
                br_monitor.add_sensor_reading(sensor_br)
        
        alpha = global_params.get_smooth_alpha()
        ema = raw if ema is None else (alpha * ema + (1 - alpha) * raw)
        warm_raw.append((raw, ema))
        hr_monitor.update()
        time.sleep(1.0 / READ_HZ)
    
    print("Warm-up done.")

    # Setup plotter and normalizer
    plotter = LivePlotter(window_size=int(PLOT_WINDOW_SECS * READ_HZ))
    norm = BreathPerCycleNormalizer(
        fs=READ_HZ,
        deriv_eps=DERIV_EPS,
        refractory_ms=REFRACTORY_MS,
        min_span=MIN_BREATH_SPAN,
        bpm_window=BPM_ROLLING
    )

    # Seed plot
    for raw, e in warm_raw[-int(READ_HZ):]:
        norm.update(e, time.time())
        plotter.add(raw, e, 0, hr_monitor.get_current_hr())

    # Initial calibration for fixed mode
    fixed_min, fixed_max = do_calibration(g, FIXED_CALIB_SECS)

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
    print(f"Calibration Mode:        {global_params.get_calib_mode().upper()}")
    print("="*60 + "\n")

    in_min, in_max = fixed_min, fixed_max
    last_sent = None
    frame = 0

    print("Streaming. Use controls to adjust. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Check for recalibration request
            if global_params.should_recalibrate():
                global_params.clear_recalibrate_flag()
                fixed_min, fixed_max = do_calibration(g, FIXED_CALIB_SECS)
                if global_params.get_calib_mode() == "fixed":
                    in_min, in_max = fixed_min, fixed_max
                    norm.set_fixed_range(in_min, in_max)
            
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]
            
            sensor_br = None
            if len(data) > 1:
                sensor_br_raw = data[1]
                if sensor_br_raw is not None:
                    sensor_br = br_monitor.add_sensor_reading(sensor_br_raw)
            
            alpha = global_params.get_smooth_alpha()
            ema = raw if ema is None else (alpha * ema + (1 - alpha) * raw)

            current_hr = hr_monitor.update()
            event, breath_duration = norm.update(ema, time.time())
            
            cycle_br = None
            if event and breath_duration is not None:
                cycle_br = br_monitor.add_breath_duration(breath_duration)
                
                mode = global_params.get_calib_mode()
                if mode == "dynamic":
                    in_min, in_max = norm.get_range()
                # For fixed mode, keep using fixed_min, fixed_max
                
                if in_min is not None and in_max is not None:
                    cycle_br_display = br_monitor.get_current_cycle_br()
                    sensor_br_display = br_monitor.get_current_sensor_br()
                    hr_display = hr_monitor.get_current_hr()
                    
                    output = f"Breath #{norm.total_breaths} → Duration: {breath_duration:.2f}s | "
                    output += f"Range: [{in_min:.5f}, {in_max:.5f}]"
                    if mode == "dynamic":
                        output += f" (avg {global_params.get_calib_window_breaths()})"
                    else:
                        output += f" (fixed)"
                    output += f"\n  BR (Cycles): {(cycle_br_display or 0):.1f} BrPM"
                    if sensor_br_display:
                        output += f" | BR (Sensor): {sensor_br_display:.1f} BrPM"
                    else:
                        output += f" | BR (Sensor): N/A"
                    output += f" | HR: {(hr_display or 0):.1f} BPM"
                    print(output)

            # Fallback if no range yet
            if in_min is None or in_max is None or in_max <= in_min:
                in_min = ema - 1e-3
                in_max = ema + 1e-3

            # Map to MIDI
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
                if br_monitor.is_cycle_ready() and cycle_br_display:
                    info = f"BR (Cycles): {cycle_br_display:.1f} BrPM"
                else:
                    breaths_needed = BR_WARMUP_BREATHS - br_monitor.breaths_detected
                    if breaths_needed > 0:
                        info = f"BR (Cycles): waiting ({breaths_needed} more breaths)"
                    else:
                        info = "BR (Cycles): stabilizing..."
                
                if br_monitor.is_sensor_ready() and sensor_br_display:
                    info += f"\nBR (Sensor): {sensor_br_display:.1f} BrPM"
                else:
                    info += f"\nBR (Sensor): stabilizing..."
                
                info += f"\nTotal breaths: {norm.total_breaths}"
                info += f"\nRange: [{in_min:.3f}, {in_max:.3f}]"
                
                if current_hr:
                    info += f"\nHeart Rate: {current_hr:.1f} BPM"
                
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
            print(f"  Breathing Rate (Cycles): Not available")
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
        if start_sensor_br and end_sensor_br:
            sensor_br_change = end_sensor_br - start_sensor_br
            print(f"  Breathing Rate Change (Sensor): {sensor_br_change:+.1f} BrPM")
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