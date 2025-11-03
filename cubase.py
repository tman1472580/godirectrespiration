"""
Go Direct Respiration Belt → Cubase Volume (MIDI CC) bridge

• macOS/Windows. Tested with mido 1.3+, python-rtmidi, Vernier gdx.
• Creates a virtual MIDI port (no IAC/loopMIDI needed). Select this port in Cubase.
• Maps belt force → smoothed 0–127 → MIDI CC (default CC7 = Volume).

Setup (one-time):
  pip install gdx mido python-rtmidi

Run:
  python cubase.py
  Then in Cubase: Studio → MIDI Remote → Input = "RespBelt CC (Python)" → Learn & bind to Volume.
"""

import time
from gdx import gdx
import mido

# ===== USER SETTINGS =====
VIRTUAL_PORT_NAME = "RespBelt CC (Python)"  # The MIDI port Cubase will see
MIDI_CHANNEL       = 1                        # 1–16
MIDI_CC            = 7                        # 7=Channel Volume; you can change this
READ_HZ       = 100      # was 20
SMOOTH_ALPHA  = 0.70     # was 0.25 (higher = less lag)
MIN_SEND_STEP = 0        # send every change                      # only send if CC changes by >= this
CALIB_SECS         = 5                        # seconds to learn min/max
SENSOR_CHANNELS    = [1]                      # Respiration belt primary channel is 1
BLE_NAME_FILTER    = "GDX-RB"                 # Tighten to your device family
# ==========================

# Use RtMidi backend and create a virtual output so Cubase can connect to it
mido.set_backend('mido.backends.rtmidi')
_midi_out = mido.open_output(VIRTUAL_PORT_NAME, virtual=True)
print(f"Created virtual MIDI port: {VIRTUAL_PORT_NAME}")


def linear_scale(x, in_min, in_max, out_min=0, out_max=127):
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return int(round(out_min + t * (out_max - out_min)))


def main():
    g = gdx.gdx()

    # Connect to the belt: try BLE first, then USB fallback
    try:
        print("wait for bluetooth initialization...")
        g.open(connection='ble')
    except Exception:
        print("BLE open failed, trying USB…")
        g.open_usb()

    # Select the sensor channel(s) we want before starting the stream
    g.select_sensors(SENSOR_CHANNELS)

    # Start streaming (period in ms)
    g.start(period=int(1000 / READ_HZ))

    # ---- Calibration ----
    print("Calibrating... breathe normally, then take a few deep breaths.")
    t0 = time.time()
    calib_vals = []
    while time.time() - t0 < CALIB_SECS:
        reading = g.read()  # list with one value per selected sensor
        if reading:
            calib_vals.append(reading[0])
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

    # ---- Streaming + MIDI ----
    ema = None
    last_sent = None
    try:
        while True:
            data = g.read()
            if not data:
                time.sleep(1.0 / READ_HZ)
                continue

            raw = data[0]
            ema = raw if ema is None else (SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * ema)
            cc_val = linear_scale(ema, in_min, in_max, 0, 127)

            if (last_sent is None) or (abs(cc_val - last_sent) >= MIN_SEND_STEP):
                msg = mido.Message('control_change', channel=MIDI_CHANNEL - 1, control=MIDI_CC, value=cc_val)
                _midi_out.send(msg)
                last_sent = cc_val

            time.sleep(1.0 / READ_HZ)

    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        g.stop()
        g.close()
        try:
            _midi_out.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
