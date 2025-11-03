# gdx_live_force.py
# Live Force (N) plot for a single Go Direct Respiration Belt (or any Go Direct with Force on channel 1)

from gdx import gdx
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import time
import math
import sys

# ===== USER CONFIG =====
DEVICE_NAME = "GDX-RB 0K700506"   # put your exact device name, or "" to choose interactively
PERIOD_MS   = 200                 # 200–500 ms is good for Force; 200 ms = 5 Hz
WINDOW_POINTS = 600               # how many recent points to show in the rolling window
# =======================

def label_from_enabled_info(info):
    """Return a single label string from g.enabled_sensor_info(), whether it is a list or a string."""
    if isinstance(info, list):
        # Use the first entry if multiple were returned
        return str(info[0]) if info else "Force (N)"
    # It's a string (some gdx versions return "['Force (N)']" as a repr)
    s = str(info).strip()
    if s.startswith('[') and s.endswith(']'):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except Exception:
            pass
    return s or "Force (N)"

def main():
    g = gdx.gdx()

    print("Connecting over BLE...")
    if DEVICE_NAME.strip():
        g.open(connection='ble', device_to_open=DEVICE_NAME)
    else:
        g.open(connection='ble')  # interactive chooser

    # Force is channel 1 for the Respiration Belt attachment
    g.select_sensors([1])         # 1D list because we are using a single device
    g.start(PERIOD_MS)

    info = g.enabled_sensor_info()
    label = label_from_enabled_info(info)
    print("\nEnabled sensors:", info)
    print("\nStreaming & plotting Force (Ctrl-C to stop)...\n")

    # --- Matplotlib live setup (single series) ---
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label=label)
    ax.set_title("Go Direct Force (live)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    xs = deque(maxlen=WINDOW_POINTS)
    ys = deque(maxlen=WINDOW_POINTS)

    t0 = time.time()
    sleep_s = PERIOD_MS / 1000.0

    try:
        while True:
            m = g.read()
            if m is None:
                print("No more data (read returned None).")
                break

            # m should be a 1-element list [force_value] for one device & one channel
            if not isinstance(m, list) or len(m) == 0:
                continue
            v = m[0]
            v = float('nan') if v is None else float(v)

            t_rel = time.time() - t0
            xs.append(t_rel)
            ys.append(v)

            line.set_data(xs, ys)

            # X limits
            if len(xs) >= 2:
                ax.set_xlim(xs[0], xs[-1] if xs[-1] - xs[0] > 1e-6 else xs[0] + 1.0)

            # Y limits (ignore NaNs)
            ys_valid = [y for y in ys if not (isinstance(y, float) and math.isnan(y))]
            if ys_valid:
                ypad = (max(ys_valid) - min(ys_valid)) * 0.1 or 1.0
                ax.set_ylim(min(ys_valid) - ypad, max(ys_valid) + ypad)

            plt.pause(0.001)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        g.stop()
        g.close()
        plt.ioff()
        plt.show()
        print("Closed.")

if __name__ == "__main__":
    main()
