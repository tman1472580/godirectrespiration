#!/usr/bin/env python3
"""
Go Direct Sensor Diagnostic Tool
This will show all available sensors and their readings
"""

from gdx import gdx
import time

def main():
    g = gdx.gdx()
    
    print("\n" + "="*60)
    print("GO DIRECT SENSOR DIAGNOSTIC")
    print("="*60)
    
    # Try to connect
    print("\nConnecting to Go Direct device...")
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
            print("❌ Could not connect to any device. Exiting.")
            return
    
    print("✓ Device connected!")
    
    # Get device info
    if hasattr(g, 'devices') and g.devices:
        device = g.devices[0]
        print(f"\nDevice: {device}")
    
    # List all available sensors
    print("\n" + "="*60)
    print("AVAILABLE SENSORS:")
    print("="*60)
    
    # Try to get sensor information without selecting them first
    try:
        # This will prompt for sensor selection and show what's available
        print("\nAttempting to list all sensors...")
        print("(This may show an interactive prompt - just wait)\n")
        
        # Don't select sensors yet, just try to see what's there
        if hasattr(g, 'discover_sensors'):
            sensors = g.discover_sensors()
            print(f"Discovered sensors: {sensors}")
    except Exception as e:
        print(f"Discovery error: {e}")
    
    # Try selecting ALL sensors to see what we get
    print("\n" + "="*60)
    print("ATTEMPTING TO SELECT ALL SENSORS")
    print("="*60)
    
    try:
        # Try selecting sensors 1-10 to see what exists
        test_sensors = list(range(1, 11))
        print(f"Trying to select sensors: {test_sensors}")
        g.select_sensors(test_sensors)
        
        # Get info about enabled sensors
        sensor_info = g.enabled_sensor_info()
        print(f"\nEnabled sensor info:\n{sensor_info}")
        
    except Exception as e:
        print(f"Error selecting all sensors: {e}")
        print("\nTrying to select sensors individually...")
        
        # Try each sensor one by one
        working_sensors = []
        for i in range(1, 11):
            try:
                g.select_sensors([i])
                working_sensors.append(i)
                info = g.enabled_sensor_info()
                print(f"✓ Sensor {i}: {info}")
            except Exception as se:
                print(f"✗ Sensor {i}: Not available ({se})")
        
        print(f"\nWorking sensors: {working_sensors}")
        
        # Now select all working sensors
        if working_sensors:
            g.select_sensors(working_sensors)
    
    # Start collecting data
    print("\n" + "="*60)
    print("READING SENSOR DATA (10 samples)")
    print("="*60)
    
    try:
        g.start(period=100)  # 100ms period
        
        sensor_info = g.enabled_sensor_info()
        print(f"\nSensor columns: {sensor_info}\n")
        
        for i in range(10):
            measurements = g.read()
            if measurements is None:
                print(f"Sample {i+1}: No data")
                time.sleep(0.1)
                continue
            
            print(f"Sample {i+1}: {measurements}")
            print(f"  Length: {len(measurements)}")
            for idx, val in enumerate(measurements):
                print(f"  Sensor {idx+1}: {val} (type: {type(val).__name__})")
            print()
            
            time.sleep(0.1)
        
    except Exception as e:
        print(f"Error reading data: {e}")
    
    # Cleanup
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    try:
        g.stop()
        g.close()
    except:
        pass
    
    print("\nDone!")

if __name__ == "__main__":
    main()