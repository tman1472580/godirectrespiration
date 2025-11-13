"""
Go Direct Respiration Belt + Polar H10 Heart Rate → Cubase Volume (MIDI CC) bridge with live plotting

• macOS/Windows. Tested with mido 1.3+, python-rtmidi, Vernier gdx.
• Creates a virtual MIDI port (no IAC/loopMIDI needed). Select this port in Cubase.
• Maps belt force → smoothed 0–127 → MIDI CC (default CC7 = Volume).
• Shows live graph of raw signal, smoothed signal, MIDI output, heart rate, and breathing rate.
• Monitors heart rate via Polar H10 Bluetooth device.
• Tracks breathing rate from sensor channel 2.
• Shows start and end metrics for both heart rate and breathing rate.

Setup (one-time):
  pip install gdx mido python-rtmidi matplotlib bleak numpy
  # Also need PolarH10.py module in the same directory

Run:
  python cubase_with_hr_br.py
  Then in Cubase: Studio → MIDI Remote → Input = "RespBelt CC (Python)" → Learn & bind to Volume.
"""

import time
import asyncio
import threading
from gdx import gdx
import mido
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from bleak import BleakScanner
from PolarH10 import PolarH10

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"
MIDI_CHANNEL       = 1
MIDI_CC            = 7
READ_HZ       = 100
SMOOTH_ALPHA  = 0.70
MIN_SEND_STEP = 0
CALIB_SECS         = 5
SENSOR_CHANNELS    = [1, 2]  # Force sensor + breathing rate sensor
BLE_NAME_FILTER    = "GDX-RB"
PLOT_WINDOW_SECS   = 10

# Polar H10 settings
POLAR_DEVICE_NAME  = "Polar H10"
HR_WARMUP_SECS     = 5
HR_START_DELAY_SECS = 10  # Capture start HR after 10 seconds

# Breathing rate settings
BR_WARMUP_SECS     = 5
BR_VALID_RANGE     = (2.0, 60.0)  # Valid breathing rate range in BrPM
BR_START_DELAY_SECS = 10  # Capture start BR after 10 seconds

# Console update interval
CONSOLE_UPDATE_INTERVAL = 2.0  # Update console every 2 seconds
# ==========================

# Use RtMidi backend and create a virtual output
mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")

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
    """
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return int(round(out_min + t * (out_max - out_min)))

class HeartRateMonitor:
    """Track heart rate metrics with thread-safe access."""
    def __init__(self):
        self.hr_readings = deque(maxlen=20)
        self.start_hr = None
        self.start_time = None
        self.first_reading_time = None
        self.polar = None
        self.lock = threading.Lock()
        self.ready = False
        self.hr_count = 0
        
    def update(self):
        """Get latest HR from Polar H10."""
        if not self.polar or not self.polar.ibi_stream_values:
            return None
            
        with self.lock:
            # Calculate instantaneous HR from most recent IBI
            recent_ibis = self.polar.ibi_stream_values[-10:]
            if recent_ibis:
                avg_ibi_ms = np.mean(recent_ibis)
                hr = 60000.0 / avg_ibi_ms  # Convert ms to BPM
                self.hr_readings.append(hr)
                self.hr_count += 1
                
                # Record time of first valid reading
                if self.first_reading_time is None:
                    self.first_reading_time = time.time()
                
                # Set start HR after 10 seconds have passed since first reading
                if self.start_hr is None and self.first_reading_time is not None:
                    elapsed = time.time() - self.first_reading_time
                    if elapsed >= HR_START_DELAY_SECS and len(self.hr_readings) >= 3:
                        self.start_hr = np.mean(list(self.hr_readings))
                        self.start_time = time.time()
                        self.ready = True
                        print(f"✓ Captured start heart rate: {self.start_hr:.1f} BPM (after {elapsed:.1f}s)")
                    
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
    
    def is_ready(self):
        """Check if we have stable heart rate data."""
        with self.lock:
            return self.ready

class BreathingRateMonitor:
    """Track breathing rate from sensor channel 2."""
    def __init__(self):
        self.br_readings = deque(maxlen=20)
        self.start_br = None
        self.start_time = None
        self.first_reading_time = None
        self.ready = False
        self.reading_count = 0
        
    def add_reading(self, br_value):
        """Add a new breathing rate reading from sensor channel 2."""
        # Validate value is reasonable
        if br_value is not None and not np.isnan(br_value) and BR_VALID_RANGE[0] <= br_value <= BR_VALID_RANGE[1]:
            self.br_readings.append(br_value)
            self.reading_count += 1
            
            # Record time of first valid reading
            if self.first_reading_time is None:
                self.first_reading_time = time.time()
            
            # Capture start value after 10 seconds have passed since first reading
            if self.start_br is None and self.first_reading_time is not None:
                elapsed = time.time() - self.first_reading_time
                if elapsed >= BR_START_DELAY_SECS and len(self.br_readings) >= 3:
                    self.start_br = np.mean(list(self.br_readings))
                    self.start_time = time.time()
                    print(f"✓ Captured start breathing rate: {self.start_br:.1f} BrPM (after {elapsed:.1f}s)")
            
            # Mark as ready after enough readings
            if not self.ready and self.reading_count >= 10:
                self.ready = True
                print(f"✓ Breathing rate monitoring stabilized")
            
            return br_value
        return None
    
    def get_current_br(self):
        """Get smoothed current breathing rate using median of recent readings."""
        if len(self.br_readings) >= 3:
            return np.median(list(self.br_readings))
        return None
    
    def get_start_br(self):
        """Get the starting breathing rate."""
        return self.start_br
    
    def is_ready(self):
        """Check if we have stable breathing rate data."""
        return self.ready

class LivePlotter:
    """Non-blocking live plot in separate thread"""
    def __init__(self, window_size):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self.raw_vals = deque(maxlen=window_size)
        self.smooth_vals = deque(maxlen=window_size)
        self.midi_vals = deque(maxlen=window_size)
        self.hr_vals = deque(maxlen=window_size)
        self.br_vals = deque(maxlen=window_size)
        
        self.fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        self.ax1, self.ax2, self.ax3, self.ax4 = axes
        self.fig.suptitle('Respiration Belt + Heart Rate → MIDI Signal Chain')
        
        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, alpha=0.3)
        
        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC Value')
        self.ax2.set_ylabel('MIDI CC (0-127)')
        self.ax2.set_ylim(-5, 132)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        self.line_hr, = self.ax3.plot([], [], 'm-', linewidth=2, label='Heart Rate')
        self.ax3.set_ylabel('Heart Rate (BPM)')
        self.ax3.set_ylim(40, 200)
        self.ax3.legend(loc='upper right')
        self.ax3.grid(True, alpha=0.3)
        
        self.line_br, = self.ax4.plot([], [], 'c-', linewidth=2, label='Breathing Rate')
        self.ax4.set_ylabel('Breathing Rate (BrPM)')
        self.ax4.set_xlabel('Time (seconds)')
        self.ax4.set_ylim(0, 60)
        self.ax4.legend(loc='upper right')
        self.ax4.grid(True, alpha=0.3)
        
        self.info_text = self.ax1.text(
            0.02, 0.98, '', transform=self.ax1.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        self.start_time = time.time()
        self.running = True
        
        plt.ion()
        plt.show()

    def add_data(self, raw, smooth, midi, hr=None, br=None):
        t = time.time() - self.start_time
        self.times.append(t)
        self.raw_vals.append(raw)
        self.smooth_vals.append(smooth)
        self.midi_vals.append(midi)
        if hr is not None:
            self.hr_vals.append(hr)
        if br is not None:
            self.br_vals.append(br)
    
    def update(self, info_text=None):
        if not self.running or not self.times:
            return
        
        times = list(self.times)
        
        # Update sensor plot
        self.line_raw.set_data(times, list(self.raw_vals))
        self.line_smooth.set_data(times, list(self.smooth_vals))
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax1.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        # Update MIDI plot
        self.line_midi.set_data(times, list(self.midi_vals))
        self.ax2.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        # Update heart rate plot
        if self.hr_vals:
            hr_times = times[-len(self.hr_vals):]
            self.line_hr.set_data(hr_times, list(self.hr_vals))
            self.ax3.relim()
            self.ax3.autoscale_view(scaley=True, scalex=False)
            self.ax3.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        # Update breathing rate plot
        if self.br_vals:
            br_times = times[-len(self.br_vals):]
            self.line_br.set_data(br_times, list(self.br_vals))
            self.ax4.relim()
            self.ax4.autoscale_view(scaley=True, scalex=False)
            self.ax4.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS), times[-1] + 0.5)
        
        if info_text:
            self.info_text.set_text(info_text)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        self.running = False
        plt.close(self.fig)

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
    g = gdx.gdx()

    # Connect to the belt
    try:
        print("wait for bluetooth initialization...")
        g.open(connection='ble')
    except Exception:
        print("BLE open failed, trying USB…")
        g.open(connection='USB')

    # Select sensors - try both channels, gracefully handle if sensor 2 fails
    has_sensor_2 = True
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
            has_sensor_2 = False
        except Exception as e2:
            print(f"Failed to select any sensors: {e2}")
            return

    g.start(period=int(1000 / READ_HZ))

    # Setup heart rate monitor in background thread
    hr_monitor = HeartRateMonitor()
    polar_thread = start_polar_thread(hr_monitor)
    
    # Setup breathing rate monitor
    br_monitor = BreathingRateMonitor()
    
    # Give Polar time to connect and warm up
    time.sleep(HR_WARMUP_SECS + 2)

    # ---- Calibration ----
    print("Calibrating... breathe normally, then take a few deep breaths.")
    t0 = time.time()
    calib_vals = []
    while time.time() - t0 < CALIB_SECS:
        reading = g.read()
        if reading:
            calib_vals.append(reading[0])
            
            # Collect breathing rate from sensor 2 if available
            if has_sensor_2 and len(reading) > 1 and reading[1] is not None:
                br_monitor.add_reading(reading[1])
            
            # Update heart rate
            hr_monitor.update()
        time.sleep(1.0 / READ_HZ)

    if not calib_vals:
        g.stop(); g.close()
        raise RuntimeError("No data received during calibration.")

    raw_min = min(calib_vals)
    raw_max = max(calib_vals)
    pad = 0.05 * (raw_max - raw_min or 1.0)
    in_min = raw_min - pad
    in_max = raw_max + pad
    print(f"Calibration done. Input range ~ [{in_min:.3f}, {in_max:.3f}]")

    # Display initial status
    print("\n" + "="*60)
    print("SESSION STARTING - Waiting for stable readings...")
    print("="*60)
    print(f"Heart Rate:       Waiting for {HR_START_DELAY_SECS}s...")
    if has_sensor_2:
        print(f"Breathing Rate:   Waiting for {BR_START_DELAY_SECS}s...")
    else:
        print(f"Breathing Rate:   Sensor not available")
    print("="*60 + "\n")

    # ---- Setup plotter ----
    max_points = int(PLOT_WINDOW_SECS * READ_HZ)
    plotter = LivePlotter(max_points)

    # ---- Streaming + MIDI ----
    ema = None
    last_sent = None
    frame_count = 0
    last_console_time = time.time()
    start_metrics_displayed = False
    
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]
            ema = raw if ema is None else (SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * ema)
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)

            # Get breathing rate from sensor 2 if available
            current_br = None
            if has_sensor_2 and len(data) > 1 and data[1] is not None:
                current_br = br_monitor.add_reading(data[1])
            
            # Update heart rate
            current_hr = hr_monitor.update()

            # Display start metrics once both are captured
            if not start_metrics_displayed:
                start_hr = hr_monitor.get_start_hr()
                start_br = br_monitor.get_start_br()
                
                # Check if we have start values (or if sensor is unavailable)
                hr_ready = start_hr is not None
                br_ready = start_br is not None or not has_sensor_2
                
                if hr_ready and br_ready:
                    print("\n" + "="*60)
                    print("SESSION START METRICS CAPTURED")
                    print("="*60)
                    if start_hr:
                        print(f"Heart Rate:       {start_hr:.1f} BPM")
                    if start_br:
                        print(f"Breathing Rate:   {start_br:.1f} BrPM")
                    elif not has_sensor_2:
                        print(f"Breathing Rate:   N/A (sensor not available)")
                    print("="*60 + "\n")
                    start_metrics_displayed = True

            # Send MIDI if changed enough
            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                msg = mido.Message('control_change', channel=MIDI_CHANNEL - 1,
                                   control=MIDI_CC, value=cc_val)
                _midi_out.send(msg)
                last_sent = cc_val

            # Update plot
            plotter.add_data(raw, ema, cc_val, current_hr, current_br)
            
            # Console output every 2 seconds
            current_time = time.time()
            if current_time - last_console_time >= CONSOLE_UPDATE_INTERVAL:
                last_console_time = current_time
                console_output = f"[{time.strftime('%H:%M:%S')}] "
                
                # Heart rate
                if current_hr:
                    console_output += f"HR: {current_hr:.1f} BPM"
                else:
                    console_output += f"HR: Detecting..."
                
                # Breathing rate
                if has_sensor_2:
                    if current_br:
                        console_output += f" | BR: {current_br:.1f} BrPM"
                    else:
                        console_output += f" | BR: Detecting..."
                else:
                    console_output += f" | BR: N/A"
                
                # MIDI CC
                console_output += f" | MIDI CC: {cc_val}"
                
                print(console_output)
            
            # Refresh plot every ~10 frames (reduces CPU load)
            frame_count += 1
            if frame_count % 10 == 0:
                info = ""
                if current_hr:
                    info += f"Heart Rate: {current_hr:.1f} BPM\n"
                else:
                    info += "Heart Rate: Detecting...\n"
                
                if has_sensor_2:
                    if current_br:
                        info += f"Breathing Rate: {current_br:.1f} BrPM\n"
                    else:
                        info += "Breathing Rate: Detecting...\n"
                else:
                    info += "Breathing Rate: N/A\n"
                
                info += f"MIDI CC: {cc_val}"
                plotter.update(info)

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("\n\nStopping…")
    finally:
        # Display end metrics
        end_hr = hr_monitor.get_current_hr()
        end_br = br_monitor.get_current_br()
        
        print("\n" + "="*60)
        print("SESSION END METRICS")
        print("="*60)
        if end_hr:
            print(f"Heart Rate:       {end_hr:.1f} BPM")
        else:
            print(f"Heart Rate:       Not available")
        if has_sensor_2:
            if end_br:
                print(f"Breathing Rate:   {end_br:.1f} BrPM")
            else:
                print(f"Breathing Rate:   Not available")
        else:
            print(f"Breathing Rate:   N/A (sensor not available)")
        print("="*60)
        
        # Get start metrics for comparison
        start_hr = hr_monitor.get_start_hr()
        start_br = br_monitor.get_start_br()
        
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print("START:")
        if start_hr:
            print(f"  Heart Rate:       {start_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:       Not available")
        if has_sensor_2:
            if start_br:
                print(f"  Breathing Rate:   {start_br:.1f} BrPM")
            else:
                print(f"  Breathing Rate:   Not available")
        else:
            print(f"  Breathing Rate:   N/A (sensor not available)")
        
        print("\nEND:")
        if end_hr:
            print(f"  Heart Rate:       {end_hr:.1f} BPM")
        else:
            print(f"  Heart Rate:       Not available")
        if has_sensor_2:
            if end_br:
                print(f"  Breathing Rate:   {end_br:.1f} BrPM")
            else:
                print(f"  Breathing Rate:   Not available")
        else:
            print(f"  Breathing Rate:   N/A (sensor not available)")
        
        print("\nCHANGES:")
        if start_hr and end_hr:
            hr_change = end_hr - start_hr
            print(f"  Heart Rate Change:       {hr_change:+.1f} BPM")
        else:
            print(f"  Heart Rate Change:       Not available")
        
        if has_sensor_2:
            if start_br and end_br:
                br_change = end_br - start_br
                print(f"  Breathing Rate Change:   {br_change:+.1f} BrPM")
            else:
                print(f"  Breathing Rate Change:   Not available")
        else:
            print(f"  Breathing Rate Change:   N/A (sensor not available)")
        print("="*60 + "\n")
        
        # Cleanup
        plotter.close()
        g.stop()
        g.close()
        try:
            _midi_out.close()
        except Exception:
            pass
        if hr_monitor.polar:
            print("Disconnecting Polar H10...")
        print("Goodbye.")

if __name__ == "__main__":
    main()