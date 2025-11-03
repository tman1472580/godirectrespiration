"""
Go Direct Respiration Belt → Cubase Volume (MIDI CC) bridge with live plotting

• macOS/Windows. Tested with mido 1.3+, python-rtmidi, Vernier gdx.
• Creates a virtual MIDI port (no IAC/loopMIDI needed). Select this port in Cubase.
• Maps belt force → smoothed 0–127 → MIDI CC (default CC7 = Volume).
• Shows live graph of raw signal, smoothed signal, and MIDI output

Setup (one-time):
  pip install gdx mido python-rtmidi matplotlib

Run:
  python cubase.py
  Then in Cubase: Studio → MIDI Remote → Input = "RespBelt CC (Python)" → Learn & bind to Volume.
"""  # Module docstring: explains what the script does and how to set it up/run it.

import time                     # Standard lib: timing, sleeping, timestamps.
from gdx import gdx             # Vernier Go Direct SDK: sensor I/O.
import mido                     # MIDI library for constructing/sending MIDI messages.
import matplotlib.pyplot as plt # For live plotting of signals.
from collections import deque   # Fast fixed-length queues for rolling windows.
import threading                # Imported but not used in this script (could be removed).

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"  # Name of virtual MIDI output the DAW will see.
MIDI_CHANNEL       = 1                       # MIDI channel (human 1–16; mido uses 0–15 internally).
MIDI_CC            = 7                       # Controller number (7 = Volume by convention).
READ_HZ       = 100                          # Sensor polling rate in Hz.
SMOOTH_ALPHA  = 0.70                         # EMA smoothing factor (closer to 1 = smoother).
MIN_SEND_STEP = 0                            # Minimum CC change to resend (0 = send every change).
CALIB_SECS         = 5                       # Seconds to collect min/max during calibration.
SENSOR_CHANNELS    = [1]                     # Go Direct sensor channel(s) to read.
BLE_NAME_FILTER    = "GDX-RB"                # (Not used here) could filter BLE devices by name.
PLOT_WINDOW_SECS   = 10  # Show last 10 seconds of data  # Time range visible in plots.
# ==========================

# Use RtMidi backend and create a virtual output
mido.set_backend('mido.backends.rtmidi')                  # Choose RtMidi backend (supports virtual ports).
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)  # Create virtual MIDI out port DAW can connect to.
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")  # Inform user which port to pick in Cubase.

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

class LivePlotter:
    """Non-blocking live plot in separate thread"""  # Class docstring: summarizes role.
    def __init__(self, window_size):
        self.window_size = window_size               # Max number of points to keep/display.
        self.times = deque(maxlen=window_size)       # Rolling buffer of timestamps.
        self.raw_vals = deque(maxlen=window_size)    # Rolling buffer of raw sensor values.
        self.smooth_vals = deque(maxlen=window_size) # Rolling buffer of EMA values.
        self.midi_vals = deque(maxlen=window_size)   # Rolling buffer of MIDI CC values.
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two stacked plots.
        self.fig.suptitle('Respiration Belt → MIDI Signal Chain')             # Figure title.
        
        self.line_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw Sensor')      # Raw line handle.
        self.line_smooth, = self.ax1.plot([], [], 'r-', linewidth=2, label='Smoothed (EMA)')  # EMA line handle.
        self.ax1.set_ylabel('Sensor Value (arbitrary units)')                 # Y label for sensor plot.
        self.ax1.legend(loc='upper right')                                    # Legend placement.
        self.ax1.grid(True, alpha=0.3)                                        # Light grid.
        
        self.line_midi, = self.ax2.plot([], [], 'g-', linewidth=2, label='MIDI CC Value')  # MIDI line handle.
        self.ax2.set_ylabel('MIDI CC (0-127)')                                 # Y label for MIDI plot.
        self.ax2.set_xlabel('Time (seconds)')                                  # Shared X label.
        self.ax2.set_ylim(-5, 132)                                             # Fixed y-limits just beyond 0–127.
        self.ax2.legend(loc='upper right')                                     # Legend placement.
        self.ax2.grid(True, alpha=0.3)                                         # Light grid.
        
        self.start_time = time.time()  # Reference time for converting absolute timestamps to seconds.
        self.running = True            # Flag to allow graceful shutdown.
        
        plt.ion()  # Turn on interactive mode so draw calls don’t block.
        plt.show() # Create/raise the window immediately.

    def add_data(self, raw, smooth, midi):
        t = time.time() - self.start_time  # Compute relative time (seconds since start).
        self.times.append(t)               # Append time to rolling buffer.
        self.raw_vals.append(raw)          # Append raw value.
        self.smooth_vals.append(smooth)    # Append smoothed value.
        self.midi_vals.append(midi)        # Append MIDI CC value.
    
    def update(self):
        if not self.running or not self.times:  # Skip if closed or nothing to draw yet.
            return
        
        times = list(self.times)  # Snapshot x-axis to a list for plotting.
        
        # Update sensor plot
        self.line_raw.set_data(times, list(self.raw_vals))         # Replace raw line data.
        self.line_smooth.set_data(times, list(self.smooth_vals))   # Replace EMA line data.
        self.ax1.relim()                                           # Recompute data limits.
        self.ax1.autoscale_view()                                  # Auto-scale y based on data.
        self.ax1.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS),    # Scroll x-window to last N seconds…
                           times[-1] + 0.5)                        # …with a small right margin.
        
        # Update MIDI plot
        self.line_midi.set_data(times, list(self.midi_vals))       # Replace MIDI line data.
        self.ax2.set_xlim(max(0, times[-1] - PLOT_WINDOW_SECS),    # Same scrolling x-window.
                           times[-1] + 0.5)
        
        self.fig.canvas.draw_idle()   # Request a draw without blocking.
        self.fig.canvas.flush_events()# Process GUI events so the plot actually updates.
    
    def close(self):
        self.running = False  # Mark as closed to stop updates.
        plt.close(self.fig)   # Close the figure window.

def main():
    g = gdx.gdx()  # Create Go Direct interface object.

    # Connect to the belt
    try:
        print("wait for bluetooth initialization...")  # Hint to user: BLE opening can take a moment.
        g.open(connection='ble')                       # Try BLE connection first.
    except Exception:
        print("BLE open failed, trying USB…")          # If BLE fails, fall back.
        g.open(connection='USB')                       # Try wired USB connection.

    g.select_sensors(SENSOR_CHANNELS)                  # Choose which sensor channel(s) to read.
    g.start(period=int(1000 / READ_HZ))                # Start streaming at desired period in ms.

    # ---- Calibration ----
    print("Calibrating... breathe normally, then take a few deep breaths.")  # User guidance.
    t0 = time.time()                 # Calibration start time.
    calib_vals = []                  # Buffer to collect calibration samples.
    while time.time() - t0 < CALIB_SECS:  # Loop for CALIB_SECS seconds.
        reading = g.read()               # Fetch latest readings (list or None).
        if reading:
            calib_vals.append(reading[0])# Store first channel’s value.
        time.sleep(1.0 / READ_HZ)        # Pace loop at READ_HZ.

    if not calib_vals:
        g.stop(); g.close()                             # Clean up if nothing came in.
        raise RuntimeError("No data received during calibration.")  # Fail fast.

    raw_min = min(calib_vals)                           # Compute observed minimum.
    raw_max = max(calib_vals)                           # Compute observed maximum.
    pad = 0.05 * (raw_max - raw_min or 1.0)             # Add 5% padding (or 5% of 1 if flat).
    in_min = raw_min - pad                              # Lower bound for scaler.
    in_max = raw_max + pad                              # Upper bound for scaler.
    print(f"Calibration done. Input range ~ [{in_min:.3f}, {in_max:.3f}]")  # Report range.

    # ---- Setup plotter ----
    max_points = int(PLOT_WINDOW_SECS * READ_HZ)  # Number of samples visible in the rolling window.
    plotter = LivePlotter(max_points)            # Create live plot manager.

    # ---- Streaming + MIDI ----
    ema = None           # Exponential moving average state (None until first sample).
    last_sent = None     # Last MIDI value we emitted (for deadband).
    frame_count = 0      # Counter to throttle UI updates.
    
    try:
        while True:                              # Main real-time loop.
            data = g.read()                      # Get most recent sample(s).
            if not data:                         # If no data this tick…
                time.sleep(1.0 / READ_HZ)        # …wait a bit and retry.
                continue

            raw = data[0]                        # First sensor channel’s raw value.
            ema = raw if ema is None else (SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * ema)  # EMA update.
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)  # Map smoothed value to MIDI range.

            # Send MIDI if changed enough
            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                msg = mido.Message('control_change', channel=MIDI_CHANNEL - 1,  # mido uses 0-based channels.
                                   control=MIDI_CC, value=cc_val)               # CC number and value.
                _midi_out.send(msg)                                             # Emit CC on virtual port.
                last_sent = cc_val                                              # Track last sent value.

            # Update plot
            plotter.add_data(raw, ema, cc_val)    # Push latest triple into buffers.
            
            # Refresh plot every ~10 frames (reduces CPU load)
            frame_count += 1                      # Bump frame counter.
            if frame_count % 10 == 0:             # Only redraw every 10th sample.
                plotter.update()                  # Redraw plot windows.

            time.sleep(1.0 / READ_HZ)             # Maintain loop timing.

    except KeyboardInterrupt:
        print("Stopping…")                        # Graceful exit on Ctrl+C.
    finally:
        plotter.close()                           # Close plot window.
        g.stop()                                  # Stop sensor streaming.
        g.close()                                  # Close device connection.
        try:
            _midi_out.close()                     # Close virtual MIDI port.
        except Exception:
            pass                                  # Ignore close errors (e.g., already closed).

if __name__ == "__main__":  # Standard “script vs import” guard.
    main()                  # Run main routine when executed directly.
