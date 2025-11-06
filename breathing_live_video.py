"""
Go Direct Respiration Belt → Video Volume Controller

Uses OpenCV for better audio/video sync - plays video with embedded audio

Setup (one-time):
  pip install gdx opencv-python pyaudio numpy moviepy

Run:
  python video_control.py
"""

import time
from gdx import gdx
import cv2
import pyaudio
import wave
import threading
import numpy as np
import tempfile
import os
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

# ===== USER SETTINGS =====
VIDEO_PATH = "/Users/tanzimmohammad/Documents/GitHub/godirectrespirationbelt/take a deep breath - Calm (720p, h264, youtube).mp4"
READ_HZ = 100
SMOOTH_ALPHA = 0.15
CALIB_SECS = 5
SENSOR_CHANNELS = [1]
# ==========================

def linear_scale(x, in_min, in_max, out_min=0, out_max=100):
    """Linear scaling maps input range to output range."""
    if in_max <= in_min:
        return out_min
    t = (x - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)

class AudioPlayer:
    """Audio player with dynamic volume control"""
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.volume = 1.0
        self.playing = False
        self.thread = None
        
        # Open audio file
        self.wf = wave.open(audio_file, 'rb')
        self.p = pyaudio.PyAudio()
        
        # Open stream
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output=True
        )
        
    def set_volume(self, volume_percent):
        """Set volume 0-100%"""
        self.volume = max(0.0, min(1.0, volume_percent / 100.0))
    
    def play(self):
        """Start audio playback in separate thread"""
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self._play_audio, daemon=True)
            self.thread.start()
    
    def _play_audio(self):
        """Audio playback loop with volume control"""
        chunk_size = 1024
        
        while self.playing:
            data = self.wf.readframes(chunk_size)
            
            if not data:
                # Loop audio
                self.wf.rewind()
                data = self.wf.readframes(chunk_size)
            
            # Apply volume
            audio_array = np.frombuffer(data, dtype=np.int16)
            audio_array = (audio_array * self.volume).astype(np.int16)
            
            self.stream.write(audio_array.tobytes())
    
    def stop(self):
        """Stop playback"""
        self.playing = False
        if self.thread:
            self.thread.join(timeout=1)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.wf.close()

class SensorReader:
    """Background thread for reading sensor data"""
    def __init__(self, gdx_instance, in_min, in_max):
        self.gdx = gdx_instance
        self.in_min = in_min
        self.in_max = in_max
        self.current_volume = 50
        self.ema = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start sensor reading thread"""
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def _read_loop(self):
        """Continuously read sensor and update volume"""
        last_print = time.time()
        
        while self.running:
            data = self.gdx.read()
            if data:
                raw = data[0]
                self.ema = raw if self.ema is None else (SMOOTH_ALPHA * raw + (1 - SMOOTH_ALPHA) * self.ema)
                self.current_volume = linear_scale(self.ema, self.in_min, self.in_max, 10, 100)
                
                # Print status occasionally
                current_time = time.time()
                if current_time - last_print > 2.0:
                    print(f"Breathing: Raw={raw:.2f}, Smoothed={self.ema:.2f}, Volume={self.current_volume:.0f}%")
                    last_print = current_time
            
            time.sleep(1.0 / READ_HZ)
    
    def get_volume(self):
        """Get current volume level"""
        return self.current_volume
    
    def stop(self):
        """Stop sensor reading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

class VideoPlayer:
    """OpenCV-based video player with synchronized audio"""
    def __init__(self, video_path):
        print("Loading video...")
        
        # Extract audio first using moviepy
        print("Extracting audio...")
        clip = VideoFileClip(video_path)
        self.fps = clip.fps
        
        # Save audio to temporary file
        self.audio_file = tempfile.mktemp(suffix='.wav')
        if clip.audio:
            clip.audio.write_audiofile(self.audio_file, verbose=False, logger=None)
        else:
            print("Warning: Video has no audio")
            self.audio_file = None
        
        clip.close()
        
        # Open video with OpenCV
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_delay = 1.0 / self.fps
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps:.1f} fps")
        print(f"Total frames: {self.frame_count}")
        
        # Setup audio player
        if self.audio_file:
            self.audio_player = AudioPlayer(self.audio_file)
        else:
            self.audio_player = None
        
        self.playing = False
        self.start_time = None
        self.frame_num = 0
        
        # Create window
        cv2.namedWindow('Breath-Controlled Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Breath-Controlled Video', self.width, self.height)
    
    def start(self):
        """Start audio playback"""
        if not self.playing:
            self.playing = True
            if self.audio_player:
                self.audio_player.play()
            self.start_time = time.time()
            self.frame_num = 0
            print("Video playback started - control volume with your breathing!")
    
    def set_volume(self, volume_percent):
        """Set audio volume 0-100%"""
        if self.audio_player:
            self.audio_player.set_volume(volume_percent)
    
    def update_and_display(self, current_volume):
        """Read and display next frame - MUST be called from main thread"""
        if not self.playing:
            return True
        
        # Calculate which frame we should be showing based on elapsed time
        elapsed = time.time() - self.start_time
        target_frame = int(elapsed * self.fps)
        
        # If we're behind, skip frames to catch up
        if target_frame > self.frame_num + 2:
            self.frame_num = target_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_time = time.time()
            self.frame_num = 0
            ret, frame = self.cap.read()
            
            if not ret:
                return False
        
        # Draw volume overlay
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (310, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Volume text
        cv2.putText(frame, f"Volume: {int(current_volume)}%", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Volume bar
        bar_x, bar_y = 20, 80
        bar_width, bar_height = 250, 30
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Volume fill
        fill_width = int((current_volume / 100.0) * bar_width)
        if current_volume > 60:
            color = (0, 255, 0)  # Green
        elif current_volume > 30:
            color = (0, 200, 255)  # Yellow
        else:
            color = (100, 100, 255)  # Red
        
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Breath-Controlled Video', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            return False
        
        self.frame_num += 1
        
        return True
    
    def close(self):
        """Cleanup resources"""
        self.playing = False
        
        if self.audio_player:
            self.audio_player.stop()
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Clean up temp audio file
        if self.audio_file and os.path.exists(self.audio_file):
            try:
                os.remove(self.audio_file)
            except:
                pass

def main():
    g = gdx.gdx()

    # Connect to the belt
    try:
        print("Wait for bluetooth initialization...")
        g.open(connection='ble')
    except Exception:
        print("BLE open failed, trying USB…")
        g.open(connection='USB')

    g.select_sensors(SENSOR_CHANNELS)
    g.start(period=int(1000 / READ_HZ))

    # ---- Calibration ----
    print("\n" + "="*50)
    print("CALIBRATION")
    print("="*50)
    print("Breathe normally for a few seconds, then take some deep breaths...")
    print(f"Collecting data for {CALIB_SECS} seconds...")
    
    t0 = time.time()
    calib_vals = []
    while time.time() - t0 < CALIB_SECS:
        reading = g.read()
        if reading:
            calib_vals.append(reading[0])
        time.sleep(1.0 / READ_HZ)

    if not calib_vals:
        g.stop()
        g.close()
        raise RuntimeError("No data received during calibration.")

    raw_min = min(calib_vals)
    raw_max = max(calib_vals)
    pad = 0.05 * (raw_max - raw_min or 1.0)
    in_min = raw_min - pad
    in_max = raw_max + pad
    
    print(f"\n✓ Calibration complete!")
    print(f"  Sensor range: [{in_min:.3f}, {in_max:.3f}]")
    print(f"  Deep breath = louder, shallow breath = quieter")
    print("="*50 + "\n")

    # ---- Setup video player and sensor reader ----
    player = VideoPlayer(VIDEO_PATH)
    sensor = SensorReader(g, in_min, in_max)
    
    # Start audio and sensor reading
    player.start()
    sensor.start()
    
    try:
        print("\n🎬 Video playing! Breathe to control the volume.")
        print("   Press ESC or 'q' to exit.\n")
        
        last_frame_time = time.time()
        
        # Main loop - just display video frames at correct rate
        while True:
            current_time = time.time()
            
            # Maintain proper frame rate
            if current_time - last_frame_time >= player.frame_delay:
                # Get current volume from sensor thread
                current_volume = sensor.get_volume()
                
                # Update audio volume
                player.set_volume(current_volume)
                
                # Display frame (MUST be in main thread)
                if not player.update_and_display(current_volume):
                    print("Video window closed")
                    break
                
                last_frame_time = current_time
            else:
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\nStopping…")
    finally:
        print("Cleaning up...")
        sensor.stop()
        player.close()
        try:
            g.stop()
        except Exception as e:
            print(f"Error stopping sensor: {e}")
        try:
            g.close()
        except Exception:
            pass
        print("Done!")

if __name__ == "__main__":
    main()