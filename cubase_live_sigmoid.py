"""
Go Direct Respiration Belt → Cubase Volume (MIDI CC) bridge with adaptive calibration
Enhanced with breath detection from force signal and breathing-synchronized recalibration

• macOS/Windows. Tested with mido 1.3+, python-rtmidi, Vernier gdx.
• Creates a virtual MIDI port (no IAC/loopMIDI needed). Select this port in Cubase.
• Maps belt force → smoothed 0–127 → MIDI CC (default CC7 = Volume).
• Shows live graph of raw signal, smoothed signal, and MIDI output
• NEW: Detects breathing rate from force sensor peaks/troughs
• NEW: Calibration window adapts to breathing rate (recalibrates every N breaths)
• NEW: Soft damping at breath peaks/troughs for natural volume transitions
• NEW: Asymmetric smoothing - fast response to shallow breathing

Setup (one-time):
  pip install gdx mido python-rtmidi matplotlib numpy

Run:
  python cubase_live_sigmoid.py
  Then in Cubase: Studio → MIDI Remote → Input = "RespBelt CC (Python)" → Learn & bind to Volume.
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
INITIAL_CALIB_SECS = 15  # Initial calibration period
BREATHS_PER_RECALIB = 1  # Recalibrate every breath (was 2)
SENSOR_CHANNELS    = [1, 2]  # 1=Force, 2=Breathing Rate (intermittent updates)
BLE_NAME_FILTER    = "GDX-RB"
PLOT_WINDOW_SECS   = 10

# BREATH DETECTION SETTINGS
BPM_WINDOW_SIZE = 20  # Number of readings to average for BPM calculation

# DAMPING SETTINGS
DAMPING_STRENGTH = 0  # Higher = more damping at extremes (1.0-5.0 recommended)
DAMPING_ZONE = 0     # Fraction of range where damping occurs (0.2 = top/bottom 20%)
# ==========================

mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")

def soft_damp_curve(x, strength=2.5, zone=0.25):
    """
    Apply soft damping at extremes (0 and 1) using a modified sigmoid curve.
    """
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
        t = (1.0 - zone) / zone if zone > 0 else 0
        t = (x - (1.0 - zone)) / zone
        eased = 1.0 / (1.0 + np.exp(-strength * (t - 0.5)))
        return (1.0 - zone) + eased * zone

def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    """Linear scaling with optional damping"""
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    
    t_damped = soft_damp_curve(t, DAMPING_STRENGTH, DAMPING_ZONE)
    
    return int(round(out_min + t_damped * (out_max - out_min)))

class BreathTracker:
    """Tracks breathing from device sensor (intermittent updates)"""
    def __init__(self, bpm_window=20):
        self.bpm_window = bpm_window
        
        # Device BPM tracking (intermittent from sensor 2)
        self.device_bpm_history = deque(maxlen=bpm_window)
        self.last_device_bpm = None
        self.last_device_update_time = time.time()
        
        # Breath counting from device BPM
        self.cumulative_time = 0
        self.total_breaths = 0
        
    def update_device_bpm(self, bpm_reading, debug=False):
        """
        Update with breath rate from device sensor (may be nan/None).
        """
        current_time = time.time()
        
        # Check if reading is valid
        if bpm_reading is not None and not np.isnan(bpm_reading) and bpm_reading > 0:
            self.last_device_bpm = bpm_reading
            self.device_bpm_history.append(bpm_reading)
            
            # Calculate breaths since last valid update
            time_delta = current_time - self.last_device_update_time
            self.last_device_update_time = current_time
            
            # Estimate breaths: (breaths/min) * (minutes elapsed)
            breaths_delta = (bpm_reading / 60.0) * time_delta
            self.cumulative_time += breaths_delta
            
            # Count whole breaths
            if self.cumulative_time >= 1.0:
                whole_breaths = int(self.cumulative_time)
                self.total_breaths += whole_breaths
                self.cumulative_time -= whole_breaths
                if debug:
                    print(f"   Breath counted! Total: {self.total_breaths} (BPM: {bpm_reading:.1f})")
    
    def get_bpm(self):
        """Get current breathing rate from device sensor"""
        if len(self.device_bpm_history) > 0:
            # Use median of recent device readings for stability
            return np.median(list(self.device_bpm_history))
        return self.last_device_bpm
    
    def get_breath_count(self):
        """Get total breath count"""
        return self.total_breaths

class AdaptiveCalibrator:
    """Manages rolling calibration that updates continuously during each breath"""
    def __init__(self, breaths_per_recalib=1, sample_rate=100, avg_breath_duration=4.0):
        self.breaths_per_recalib = breaths_per_recalib
        self.sample_rate = sample_rate
        self.last_breath_count = 0
        
        # Rolling buffer that holds exactly N complete breaths
        # Smaller buffer = faster response to changes
        buffer_size = int(breaths_per_recalib * avg_breath_duration * sample_rate * 1.2)
        self.calib_buffer = deque(maxlen=buffer_size)
        
        # Separate tracking for per-breath statistics
        self.current_breath_samples = []
        self.last_breath_min = None
        self.last_breath_max = None
        
        self.in_min = None
        self.in_max = None
        self.smoothing_alpha_up = 0.2    # Smooth when range increases (slower)
        self.smoothing_alpha_down = 0.95  # Very fast response when range decreases (highly sensitive)
        
    def add_sample(self, value):
        """Add a sample to both rolling buffer and current breath tracker"""
        self.calib_buffer.append(value)
        self.current_breath_samples.append(value)
        
    def on_breath_complete(self):
        """Called when a breath completes - update per-breath statistics"""
        if len(self.current_breath_samples) > 10:
            self.last_breath_min = min(self.current_breath_samples)
            self.last_breath_max = max(self.current_breath_samples)
        self.current_breath_samples.clear()
    
    def should_recalibrate(self, current_breath_count):
        """Check if we should update calibration (now every breath)"""
        breaths_since_last = current_breath_count - self.last_breath_count
        return breaths_since_last >= self.breaths_per_recalib
    
    def recalibrate(self, current_breath_count):
        """Update min/max using rolling buffer - this gives us the range from recent breaths"""
        if len(self.calib_buffer) < 50:
            return False
        
        # Calculate new range from rolling buffer
        raw_min = min(self.calib_buffer)
        raw_max = max(self.calib_buffer)
        pad = 0.03 * (raw_max - raw_min or 1.0)  # Smaller padding for tighter range
        
        new_min = raw_min - pad
        new_max = raw_max + pad
        
        # Smooth the calibration changes to avoid sudden jumps
        # BUT: respond faster when range is decreasing (breathing gets shallower)
        if self.in_min is None:
            self.in_min = new_min
            self.in_max = new_max
        else:
            # For minimum: use fast response when going up, slow when going down
            if new_min > self.in_min:
                # Minimum increasing (less force at rest) - respond quickly
                alpha_min = self.smoothing_alpha_down
            else:
                # Minimum decreasing - respond slowly
                alpha_min = self.smoothing_alpha_up
            
            # For maximum: use fast response when going DOWN, slow when going up
            if new_max < self.in_max:
                # Maximum decreasing (shallower breathing) - respond quickly!
                alpha_max = self.smoothing_alpha_down
            else:
                # Maximum increasing (deeper breathing) - respond slowly
                alpha_max = self.smoothing_alpha_up
            
            self.in_min = alpha_min * new_min + (1 - alpha_min) * self.in_min
            self.in_max = alpha_max * new_max + (1 - alpha_max) * self.in_max
        
        self.last_breath_count = current_breath_count
        
        return True
    
    def get_range(self):
        """Get current calibration range"""
        return self.in_min, self.in_max

class LivePlotter:
    """Non-blocking live plot in separate thread"""
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
        
        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC (damped)')
        self.ax2.set_ylabel('MIDI CC (0-127)')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylim(-5, 132)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        # Info text
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

    # Connect to the belt with better error handling
    print("\n=== DEVICE CONNECTION ===")
    print("Attempting to connect to Go Direct Respiration Belt...")
    print("\nTROUBLESHOOTING TIPS:")
    print("1. Make sure the belt is powered on (LED should be visible)")
    print("2. Ensure Bluetooth is enabled on your computer")
    print("3. Try opening Graphical Analysis app first to verify belt works")
    print("4. If belt was recently connected, try turning it off/on")
    print("5. Check that no other app is connected to the belt\n")
    
    connected = False
    device = None
    
    # Try BLE with proper device list checking
    try:
        print("Trying Bluetooth Low Energy (BLE) connection...")
        result = g.open(connection='ble')
        
        # Wait a moment for device list to populate
        time.sleep(1.0)
        
        # Check if we have valid devices
        if g.devices and len(g.devices) > 0:
            # Filter for respiration belt or just use first device
            for dev in g.devices:
                if dev:  # Check device is not None/False
                    device = dev
                    connected = True
                    print(f"✓ BLE connection successful!")
                    print(f"  Device: {device}")
                    break
        
        if not connected:
            print("✗ BLE connection failed (no valid devices in list)")
            print(f"  Debug: g.devices = {g.devices}")
            
    except Exception as e:
        print(f"✗ BLE connection failed: {e}")
    
    # Try USB if BLE failed
    if not connected:
        try:
            print("\nTrying USB connection...")
            result = g.open(connection='usb')
            time.sleep(0.5)
            
            if g.devices and len(g.devices) > 0:
                for dev in g.devices:
                    if dev:
                        device = dev
                        connected = True
                        print("✓ USB connection successful!")
                        print(f"  Device: {device}")
                        break
            
            if not connected:
                print("✗ USB connection failed (no valid devices in list)")
                
        except Exception as e:
            print(f"✗ USB connection failed: {e}")
    
    if not connected or not device:
        print("\n❌ ERROR: Could not connect to any device.")
        print("\nPlease check:")
        print("- Device is powered on and LED is flashing")
        print("- Bluetooth is enabled (for wireless)")
        print("- Device is not connected to another application")
        print("- Try power cycling the device (off/on)")
        print("- Try opening Vernier Graphical Analysis to verify device works")
        print("\nIf GA works but this script doesn't, the device may need a firmware update.")
        return

    print(f"\n✓ Successfully connected!")

    # List available sensors
    print("\n=== AVAILABLE SENSORS ===")
    try:
        sensors = device.list_sensors()
        if sensors and isinstance(sensors, dict):
            for sensor_num, sensor_info in sensors.items():
                print(f"  Sensor {sensor_num}: {sensor_info}")
        else:
            print("  Warning: No sensors listed (this may be normal)")
            print(f"  Debug: sensors = {sensors}")
    except Exception as e:
        print(f"  Could not list sensors: {e}")
        print("  (Continuing anyway - this may not be critical)")
    
    # Select sensors with error handling
    print(f"\n=== SENSOR SELECTION ===")
    selected_sensors = SENSOR_CHANNELS  # Local copy
    print(f"Attempting to select sensors: {selected_sensors}")
    try:
        g.select_sensors(selected_sensors)
        print("✓ Sensors selected successfully")
    except Exception as e:
        print(f"✗ Failed to select sensors {selected_sensors}: {e}")
        print("\nTrying to select only sensor 1 (Force)...")
        try:
            g.select_sensors([1])
            selected_sensors = [1]  # Update to reflect actual selection
            print("✓ Successfully selected sensor 1 only")
        except Exception as e2:
            print(f"✗ Failed to select sensor 1: {e2}")
            g.stop()
            g.close()
            return
    
    # Start data collection
    print(f"\n=== STARTING DATA COLLECTION ===")
    try:
        g.start(period=int(1000 / READ_HZ))
        print(f"✓ Data collection started at {READ_HZ} Hz")
    except Exception as e:
        print(f"✗ Failed to start data collection: {e}")
        g.close()
        return

    # Test data collection before calibration
    print("\nTesting data collection...")
    test_samples = 0
    test_start = time.time()
    while test_samples < 10 and time.time() - test_start < 5:
        reading = g.read()
        if reading and len(reading) >= 1:
            test_samples += 1
            if test_samples == 1:
                print(f"  First reading: {reading}")
        time.sleep(0.1)
    
    if test_samples == 0:
        print("✗ No data received from device!")
        print("  The device may not be streaming properly.")
        g.stop()
        g.close()
        return
    
    print(f"✓ Received {test_samples} test samples")

    # ---- Initial Calibration ----
    print(f"\n=== INITIAL CALIBRATION ({INITIAL_CALIB_SECS}s) ===")
    print("Breathe normally, then take a few deep breaths...")
    t0 = time.time()
    calib_vals = []
    while time.time() - t0 < INITIAL_CALIB_SECS:
        reading = g.read()
        if reading and len(reading) >= 1:
            calib_vals.append(reading[0])  # Force sensor
            if len(calib_vals) % 100 == 0:
                print(f"  Collected {len(calib_vals)} samples...")
        time.sleep(1.0 / READ_HZ)

    if not calib_vals:
        print("\n❌ ERROR: No data received during calibration.")
        print("The sensor may not be working properly.")
        g.stop()
        g.close()
        return

    # Initialize adaptive calibrator and breath tracker
    calibrator = AdaptiveCalibrator(BREATHS_PER_RECALIB, READ_HZ)
    breath_tracker = BreathTracker(BPM_WINDOW_SIZE)
    
    for val in calib_vals:
        calibrator.add_sample(val)
    calibrator.recalibrate(0)
    
    in_min, in_max = calibrator.get_range()
    print(f"✓ Initial calibration complete!")
    print(f"  Samples collected: {len(calib_vals)}")
    print(f"  Range: [{in_min:.3f}, {in_max:.3f}]")
    print(f"  Adaptive recalibration: every {BREATHS_PER_RECALIB} breaths")
    print(f"  Damping: strength={DAMPING_STRENGTH}, zone={DAMPING_ZONE*100:.0f}%")
    print(f"  Asymmetric smoothing: up={calibrator.smoothing_alpha_up}, down={calibrator.smoothing_alpha_down}")
    
    if len(selected_sensors) > 1:
        print(f"  Breath detection: using device sensor (channel 2)")
    else:
        print(f"  Note: Only force sensor available (breath rate won't update)")

    # ---- Setup plotter ----
    max_points = int(PLOT_WINDOW_SECS * READ_HZ)
    plotter = LivePlotter(max_points)

    # ---- Streaming + MIDI ----
    print(f"\n=== STREAMING DATA ===")
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

            # Extract force and breathing rate readings
            raw_force = data[0] if len(data) > 0 else None
            breathing_rate = data[1] if len(data) > 1 else None
            
            if raw_force is None:
                time.sleep(1.0 / READ_HZ)
                continue
            
            current_time = time.time()
            
            # Update breath tracker with device BPM (handles nan gracefully)
            if breathing_rate is not None:
                breath_tracker.update_device_bpm(breathing_rate, debug=True)
            
            # Smooth force signal
            ema = raw_force if ema is None else (SMOOTH_ALPHA * raw_force + (1 - SMOOTH_ALPHA) * ema)
            
            # Add to calibration buffer
            calibrator.add_sample(raw_force)
            
            # Get current breath count
            current_breath_count = breath_tracker.get_breath_count()
            
            # Check if recalibration is needed (based on breath count)
            if calibrator.should_recalibrate(current_breath_count):
                print(f"   Recalibration triggered: breath count = {current_breath_count}, last = {calibrator.last_breath_count}")
                calibrator.recalibrate(current_breath_count)
                in_min, in_max = calibrator.get_range()
                recalib_count += 1
                bpm = breath_tracker.get_bpm()
                bpm_str = f"{bpm:.1f}" if bpm else "calculating..."
                print(f" Recalibrated #{recalib_count} after {BREATHS_PER_RECALIB} breaths | "
                      f"Breathing rate: {bpm_str} BPM | Range: [{in_min:.3f}, {in_max:.3f}]")
            
            # Get current range
            in_min, in_max = calibrator.get_range()
            
            # Scale to MIDI
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)

            # Send MIDI if changed enough
            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                msg = mido.Message('control_change', channel=MIDI_CHANNEL - 1,
                                   control=MIDI_CC, value=cc_val)
                _midi_out.send(msg)
                last_sent = cc_val

            # Update plot
            plotter.add_data(raw_force, ema, cc_val)
            
            frame_count += 1
            if frame_count % 10 == 0:
                # Prepare info text
                bpm = breath_tracker.get_bpm()
                bpm_str = f"{bpm:.1f} BPM" if bpm else "Detecting breaths..."
                breaths_until_recalib = BREATHS_PER_RECALIB - (current_breath_count - calibrator.last_breath_count)
                
                info_text = (f"Breathing Rate: {bpm_str}\n"
                            f"Total breaths: {current_breath_count} | Next recalib in: {breaths_until_recalib} breaths\n"
                            f"Range: [{in_min:.2f}, {in_max:.2f}]")
                plotter.update(info_text)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\n\n=== STOPPING ===")
        bpm = breath_tracker.get_bpm()
        if bpm:
            print(f"Final breathing rate: {bpm:.1f} BPM ({current_breath_count} total breaths)")
        print("Session complete!")
    finally:
        plotter.close()
        g.stop()
        g.close()
        try:
            _midi_out.close()
        except Exception:
            pass
        print("Device disconnected. Goodbye!")

if __name__ == "__main__":
    main()