"""
Go Direct Respiration Belt → Cubase with NORMALIZATION
Ensures consistent 0-127 MIDI range regardless of breath depth

Normalization modes:
  'adaptive'   - Your current approach (recalibrates every N breaths)
  'per_breath' - Each breath uses full 0-127 range
  'rolling'    - Normalize to last N seconds of data
  'percentile' - Use 5th/95th percentiles (most robust to outliers)
"""

import time
from gdx import gdx
import mido
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"
MIDI_CHANNEL       = 1
MIDI_CC            = 7
READ_HZ       = 100
SMOOTH_ALPHA  = 0.95
MIN_SEND_STEP = 0
INITIAL_CALIB_SECS = 15
BREATHS_PER_RECALIB = 2
SENSOR_CHANNELS    = [1, 2]
PLOT_WINDOW_SECS   = 10

# NORMALIZATION SETTINGS
NORMALIZATION_MODE = "per_breath"  # Options: 'adaptive', 'per_breath', 'rolling', 'percentile'
ROLLING_WINDOW_SECS = 10  # For 'rolling' and 'percentile' modes
PERCENTILE_LOW = 5   # For 'percentile' mode
PERCENTILE_HIGH = 95 # For 'percentile' mode

# DAMPING SETTINGS
DAMPING_STRENGTH = 4.5
DAMPING_ZONE = 0.25

# BREATH DETECTION
BPM_WINDOW_SIZE = 20
# ==========================

mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")
print(f"Normalization mode: {NORMALIZATION_MODE}")

def soft_damp_curve(x, strength=2.5, zone=0.25):
    """Apply soft damping at extremes"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    dist_from_bottom = x
    dist_from_top = 1.0 - x
    min_dist = min(dist_from_bottom, dist_from_top)
    
    if min_dist > zone:
        return x
    
    if x < 0.5:
        t = x / zone
        eased = 1.0 / (1.0 + np.exp(-strength * (t - 0.5)))
        return eased * zone
    else:
        t = (x - (1.0 - zone)) / zone
        eased = 1.0 / (1.0 + np.exp(-strength * (t - 0.5)))
        return (1.0 - zone) + eased * zone

def linear_scale(x, in_min, in_max, out_min=0, out_max=127, use_damping=True):
    """Linear scaling with optional damping"""
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    
    if use_damping:
        t = soft_damp_curve(t, DAMPING_STRENGTH, DAMPING_ZONE)
    
    return int(round(out_min + t * (out_max - out_min)))

class PerBreathNormalizer:
    """Normalize each breath cycle to full 0-127 range"""
    def __init__(self, settle_samples=20):
        self.current_min = None
        self.current_max = None
        self.prev_value = None
        self.direction = 0
        self.settle_samples = settle_samples
        self.samples_in_cycle = 0
        
    def update(self, value):
        if self.prev_value is None:
            self.prev_value = value
            self.current_min = value
            self.current_max = value
            return 64
        
        # Detect direction change
        delta = value - self.prev_value
        threshold = 0.01
        new_direction = 1 if delta > threshold else (-1 if delta < -threshold else self.direction)
        
        # Reset on direction change
        if new_direction != self.direction and new_direction != 0:
            self.current_min = value
            self.current_max = value
            self.direction = new_direction
            self.samples_in_cycle = 0
        
        # Track extremes
        self.current_min = min(self.current_min, value)
        self.current_max = max(self.current_max, value)
        self.samples_in_cycle += 1
        
        # Normalize
        if self.current_max > self.current_min and self.samples_in_cycle > self.settle_samples:
            t = (value - self.current_min) / (self.current_max - self.current_min)
            t = max(0.0, min(1.0, t))
            t = soft_damp_curve(t, DAMPING_STRENGTH, DAMPING_ZONE)
            midi = int(t * 127)
        else:
            midi = 64
        
        self.prev_value = value
        return midi

class RollingNormalizer:
    """Normalize based on sliding window"""
    def __init__(self, window_seconds, sample_rate):
        self.window = deque(maxlen=window_seconds * sample_rate)
        
    def update(self, value):
        self.window.append(value)
        
        if len(self.window) < 20:
            return 64
        
        window_min = min(self.window)
        window_max = max(self.window)
        
        return linear_scale(value, window_min, window_max, 0, 127)

class PercentileNormalizer:
    """Normalize using percentiles (robust to outliers)"""
    def __init__(self, window_seconds, sample_rate, low_pct=5, high_pct=95):
        self.window = deque(maxlen=window_seconds * sample_rate)
        self.low_pct = low_pct
        self.high_pct = high_pct
        
    def update(self, value):
        self.window.append(value)
        
        if len(self.window) < 50:
            return 64
        
        # Calculate percentiles
        window_list = list(self.window)
        p_low = np.percentile(window_list, self.low_pct)
        p_high = np.percentile(window_list, self.high_pct)
        
        return linear_scale(value, p_low, p_high, 0, 127)

class BreathTracker:
    """Tracks breathing from device sensor"""
    def __init__(self, bpm_window=20):
        self.bpm_window = bpm_window
        self.device_bpm_history = deque(maxlen=bpm_window)
        self.last_device_bpm = None
        self.last_device_update_time = time.time()
        self.cumulative_time = 0
        self.total_breaths = 0
        
    def update_device_bpm(self, bpm_reading, debug=False):
        current_time = time.time()
        
        if bpm_reading is not None and not np.isnan(bpm_reading) and bpm_reading > 0:
            self.last_device_bpm = bpm_reading
            self.device_bpm_history.append(bpm_reading)
            
            time_delta = current_time - self.last_device_update_time
            self.last_device_update_time = current_time
            
            breaths_delta = (bpm_reading / 60.0) * time_delta
            self.cumulative_time += breaths_delta
            
            if self.cumulative_time >= 1.0:
                whole_breaths = int(self.cumulative_time)
                self.total_breaths += whole_breaths
                self.cumulative_time -= whole_breaths
                if debug:
                    print(f"   Breath counted! Total: {self.total_breaths} (BPM: {bpm_reading:.1f})")
    
    def get_bpm(self):
        if len(self.device_bpm_history) > 0:
            return np.median(list(self.device_bpm_history))
        return self.last_device_bpm
    
    def get_breath_count(self):
        return self.total_breaths

class AdaptiveCalibrator:
    """Adaptive recalibration (original mode)"""
    def __init__(self, breaths_per_recalib=3, sample_rate=100, avg_breath_duration=4.0):
        self.breaths_per_recalib = breaths_per_recalib
        self.sample_rate = sample_rate
        self.last_breath_count = 0
        
        buffer_size = int(breaths_per_recalib * avg_breath_duration * sample_rate * 1.2)
        self.calib_buffer = deque(maxlen=buffer_size)
        
        self.window_start_breath_count = 0
        self.in_min = None
        self.in_max = None
        
    def add_sample(self, value):
        self.calib_buffer.append(value)
        
    def should_recalibrate(self, current_breath_count):
        breaths_since_last = current_breath_count - self.last_breath_count
        return breaths_since_last >= self.breaths_per_recalib
    
    def recalibrate(self, current_breath_count):
        if len(self.calib_buffer) < 50:
            return False
        
        raw_min = min(self.calib_buffer)
        raw_max = max(self.calib_buffer)
        pad = 0.05 * (raw_max - raw_min or 1.0)
        
        self.in_min = raw_min - pad
        self.in_max = raw_max + pad
        self.last_breath_count = current_breath_count
        
        self.calib_buffer.clear()
        self.window_start_breath_count = current_breath_count
        
        return True
    
    def update(self, value):
        """For compatibility with other normalizers"""
        return linear_scale(value, self.in_min, self.in_max, 0, 127)
    
    def get_range(self):
        return self.in_min, self.in_max

class LivePlotter:
    """Non-blocking live plot"""
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.raw_vals = deque(maxlen=window_size)
        self.smooth_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle(f'Respiration Belt → MIDI ({NORMALIZATION_MODE.upper()} normalization)')
        
        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC (normalized)')
        self.ax2.set_ylabel('MIDI CC (0-127)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylim(-5, 132)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        self.info_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                       verticalalignment='top', fontsize=9,
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.start_time = time.time()
        self.running = True
        
        plt.ion()
        plt.show()

    def add_data(self, raw, smooth, midi):
        t = time.time() - self.start_time
        self.times.append(t)
        self.raw_vals.append(raw)
        self.smooth_vals.append(smooth)
        self.midi_vals.append(midi)
    
    def update(self, info_text=None):
        if not self.running or not self.times:
            return
        
        times = list(self.times)
        
        self.line_raw.set_data(times, list(self.raw_vals))
        self.line_smooth.set_data(times, list(self.smooth_vals))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax1.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        self.line_midi.set_data(times, list(self.midi_vals))
        self.ax2.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        if info_text:
            self.info_text.set_text(info_text)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        self.running = False
        plt.close(self.fig)

def main():
    g = gdx.gdx()

    print("\n=== DEVICE CONNECTION ===")
    print("Connecting to Go Direct Respiration Belt...")
    
    connected = False
    
    try:
        print("Trying Bluetooth (BLE)...")
        g.open(connection='ble')
        time.sleep(1.0)
        if g.devices and len(g.devices) > 0:
            connected = True
            print("✓ BLE connected!")
    except Exception as e:
        print(f"✗ BLE failed: {e}")
    
    if not connected:
        try:
            print("Trying USB...")
            g.open(connection='usb')
            time.sleep(0.5)
            if g.devices and len(g.devices) > 0:
                connected = True
                print("✓ USB connected!")
        except Exception as e:
            print(f"✗ USB failed: {e}")
    
    if not connected:
        print("\n❌ Could not connect to device!")
        return

    # Select sensors
    selected_sensors = SENSOR_CHANNELS
    try:
        g.select_sensors(selected_sensors)
        print(f"✓ Sensors selected: {selected_sensors}")
    except Exception:
        print("⚠ Selecting sensor 1 only...")
        g.select_sensors([1])
        selected_sensors = [1]
    
    # Start collection
    g.start(period=int(1000 / READ_HZ))
    print(f"✓ Started at {READ_HZ} Hz")

    # Initial calibration
    print(f"\n=== CALIBRATION ({INITIAL_CALIB_SECS}s) ===")
    print("Breathe normally, then deep...")
    t0 = time.time()
    calib_vals = []
    while time.time() - t0 < INITIAL_CALIB_SECS:
        reading = g.read()
        if reading and len(reading) >= 1:
            calib_vals.append(reading[0])
        time.sleep(1.0 / READ_HZ)

    if not calib_vals:
        print("❌ No calibration data!")
        g.stop()
        g.close()
        return

    print(f"✓ Collected {len(calib_vals)} samples")

    # Initialize normalizer based on mode
    if NORMALIZATION_MODE == 'adaptive':
        normalizer = AdaptiveCalibrator(BREATHS_PER_RECALIB, READ_HZ)
        for val in calib_vals:
            normalizer.add_sample(val)
        normalizer.recalibrate(0)
        print(f"  Mode: Adaptive (recalibrate every {BREATHS_PER_RECALIB} breaths)")
    elif NORMALIZATION_MODE == 'per_breath':
        normalizer = PerBreathNormalizer()
        print(f"  Mode: Per-breath (each breath uses full 0-127)")
    elif NORMALIZATION_MODE == 'rolling':
        normalizer = RollingNormalizer(ROLLING_WINDOW_SECS, READ_HZ)
        print(f"  Mode: Rolling window ({ROLLING_WINDOW_SECS}s)")
    elif NORMALIZATION_MODE == 'percentile':
        normalizer = PercentileNormalizer(ROLLING_WINDOW_SECS, READ_HZ, PERCENTILE_LOW, PERCENTILE_HIGH)
        print(f"  Mode: Percentile ({PERCENTILE_LOW}th-{PERCENTILE_HIGH}th percentile)")
    else:
        print(f"❌ Unknown mode: {NORMALIZATION_MODE}")
        return

    breath_tracker = BreathTracker(BPM_WINDOW_SIZE)
    max_points = int(PLOT_WINDOW_SECS * READ_HZ)
    plotter = LivePlotter(max_points)

    print(f"\n=== STREAMING ===")
    print("Press Ctrl+C to stop\n")
    
    ema = None
    last_sent = None
    frame_count = 0
    recalib_count = 0
    
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw_force = data[0] if len(data) > 0 else None
            breathing_rate = data[1] if len(data) > 1 else None
            
            if raw_force is None:
                time.sleep(1.0 / READ_HZ)
                continue
            
            # Update breath tracker
            if breathing_rate is not None:
                breath_tracker.update_device_bpm(breathing_rate, debug=True)
            
            # Smooth signal
            ema = raw_force if ema is None else (SMOOTH_ALPHA * raw_force + (1 - SMOOTH_ALPHA) * ema)
            
            # Get normalized MIDI value
            if NORMALIZATION_MODE == 'adaptive':
                normalizer.add_sample(raw_force)
                current_breath_count = breath_tracker.get_breath_count()
                
                if normalizer.should_recalibrate(current_breath_count):
                    normalizer.recalibrate(current_breath_count)
                    recalib_count += 1
                    in_min, in_max = normalizer.get_range()
                    bpm = breath_tracker.get_bpm()
                    print(f"  Recalibrated #{recalib_count} | BPM: {bpm:.1f if bpm else 'N/A'} | Range: [{in_min:.3f}, {in_max:.3f}]")
                
                cc_val = normalizer.update(ema)
            else:
                cc_val = normalizer.update(ema)

            # Send MIDI
            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                msg = mido.Message('control_change', channel=MIDI_CHANNEL - 1,
                                   control=MIDI_CC, value=cc_val)
                _midi_out.send(msg)
                last_sent = cc_val

            # Update plot
            plotter.add_data(raw_force, ema, cc_val)
            
            frame_count += 1
            if frame_count % 10 == 0:
                bpm = breath_tracker.get_bpm()
                bpm_str = f"{bpm:.1f} BPM" if bpm else "Detecting..."
                
                info_text = f"Mode: {NORMALIZATION_MODE}\nBreathing: {bpm_str}\nTotal breaths: {breath_tracker.get_breath_count()}"
                
                if NORMALIZATION_MODE == 'adaptive':
                    current_breath_count = breath_tracker.get_breath_count()
                    breaths_until = BREATHS_PER_RECALIB - (current_breath_count - normalizer.last_breath_count)
                    info_text += f"\nNext recalib: {breaths_until} breaths"
                
                plotter.update(info_text)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\n\n=== STOPPED ===")
        print("Session complete!")
    finally:
        plotter.close()
        g.stop()
        g.close()
        try:
            _midi_out.close()
        except Exception:
            pass
        print("Goodbye!")

if __name__ == "__main__":
    main()