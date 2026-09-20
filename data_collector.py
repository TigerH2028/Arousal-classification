"""
FILE: data_collector.py
PURPOSE: Collects multimodal data from all sensors simultaneously.
         Runs on your PC during VR sessions.
         
HOW TO RUN:
  python data_collector.py --participant P001 --session 1 --condition BASELINE

REQUIRES: Arduino connected via USB, Polar H10 via Bluetooth, webcam plugged in.
"""

import asyncio
import serial
import serial.tools.list_ports
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import argparse
import queue
from datetime import datetime
import bleak  # For Bluetooth (Polar H10)


# ============================================================
# CONFIGURATION
# ============================================================

ARDUINO_BAUD = 115200
GSR_SAMPLE_HZ = 20          # Must match Arduino sketch
CAMERA_FPS = 15             # Webcam capture rate
DATA_FOLDER = "data/raw"    # Where to save files

# Polar H10 Bluetooth UUIDs (standard for BLE HR sensors)
POLAR_HR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
POLAR_ECG_UUID = "fb005c82-02e7-f387-1cad-8acd2d8df0c8"


# ============================================================
# HELPER: Find Arduino Port Automatically
# ============================================================

def find_arduino_port():
    """Scans all COM ports and returns the one with Arduino."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description or "USB Serial" in port.description:
            print(f"  Found Arduino on port: {port.device}")
            return port.device
    
    # If not found automatically, list all and ask user
    print("Could not find Arduino automatically. Available ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device}: {port.description}")
    idx = int(input("Enter port number: "))
    return ports[idx].device


# ============================================================
# SENSOR THREAD 1: GSR (via Arduino)
# ============================================================

class GSRCollector(threading.Thread):
    """
    Reads GSR data from Arduino in a background thread.
    Writes to a thread-safe queue.
    """
    
    def __init__(self, data_queue, stop_event):
        super().__init__(daemon=True)
        self.queue = data_queue
        self.stop_event = stop_event
        self.port = None
        self.connected = False
    
    def connect(self):
        port_name = find_arduino_port()
        self.port = serial.Serial(port_name, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Read header lines
        for _ in range(3):
            line = self.port.readline().decode('utf-8', errors='ignore').strip()
            print(f"  Arduino: {line}")
        
        self.connected = True
        print("  GSR sensor connected.")
    
    def run(self):
        if not self.connected:
            print("  WARNING: GSR not connected, skipping.")
            return
        
        while not self.stop_event.is_set():
            try:
                line = self.port.readline().decode('utf-8', errors='ignore').strip()
                if ',' in line and not line.startswith('TIMESTAMP'):
                    parts = line.split(',')
                    if len(parts) == 4:
                        record = {
                            'timestamp_ms': int(parts[0]),
                            'system_time': time.time(),
                            'sensor': 'gsr',
                            'raw_adc': int(parts[1]),
                            'voltage': float(parts[2]),
                            'conductance_us': float(parts[3])
                        }
                        self.queue.put(record)
            except Exception as e:
                pass  # Skip bad lines silently
    
    def close(self):
        if self.port and self.port.is_open:
            self.port.close()


# ============================================================
# SENSOR THREAD 2: Heart Rate (Polar H10 via Bluetooth)
# ============================================================

class PolarH10Collector:
    """
    Connects to Polar H10 via BLE and streams HR + RR intervals (for HRV).
    RR intervals are the gaps between heartbeats in milliseconds.
    """
    
    def __init__(self, data_queue, stop_event):
        self.queue = data_queue
        self.stop_event = stop_event
        self.device_address = None
    
    def parse_hr_measurement(self, data):
        """Parse BLE Heart Rate Measurement characteristic."""
        flag = data[0]
        hr_format = flag & 0x01  # 0 = UINT8, 1 = UINT16
        
        if hr_format:
            heart_rate = int.from_bytes(data[1:3], 'little')
            rr_start = 3
        else:
            heart_rate = data[1]
            rr_start = 2
        
        # RR intervals (time between beats in 1/1024 seconds)
        rr_intervals = []
        for i in range(rr_start, len(data) - 1, 2):
            rr_raw = int.from_bytes(data[i:i+2], 'little')
            rr_ms = (rr_raw / 1024.0) * 1000.0  # Convert to milliseconds
            rr_intervals.append(round(rr_ms, 1))
        
        return heart_rate, rr_intervals
    
    def notification_handler(self, sender, data):
        hr, rr_list = self.parse_hr_measurement(data)
        record = {
            'system_time': time.time(),
            'sensor': 'polar_h10',
            'heart_rate_bpm': hr,
            'rr_intervals_ms': rr_list,
            'rr_mean_ms': np.mean(rr_list) if rr_list else None
        }
        self.queue.put(record)
    
    async def run_async(self):
        """Scan for Polar H10 and stream data."""
        print("  Scanning for Polar H10...")
        
        devices = await bleak.BleakScanner.discover(timeout=10.0)
        polar_devices = [d for d in devices if d.name and 'Polar' in d.name]
        
        if not polar_devices:
            print("  WARNING: No Polar device found. Is it on and charged?")
            return
        
        device = polar_devices[0]
        self.device_address = device.address
        print(f"  Found: {device.name} ({device.address})")
        
        async with bleak.BleakClient(device.address) as client:
            await client.start_notify(POLAR_HR_UUID, self.notification_handler)
            print("  Polar H10 streaming HR + RR intervals.")
            
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)
            
            await client.stop_notify(POLAR_HR_UUID)
    
    def run(self):
        asyncio.run(self.run_async())


# ============================================================
# SENSOR THREAD 3: Facial Action Units (Webcam + MediaPipe)
# ============================================================

class FaceCollector(threading.Thread):
    """
    Captures webcam frames and extracts facial landmarks using MediaPipe.
    These landmarks map to Facial Action Units (AUs) related to emotion.
    """
    
    # Key landmarks for emotion-relevant regions
    LANDMARKS = {
        'brow_inner_left': 107,    # AU1: Inner brow raise
        'brow_inner_right': 336,
        'brow_outer_left': 46,     # AU2: Outer brow raise
        'brow_outer_right': 276,
        'brow_lowerer_left': 66,   # AU4: Brow lowerer (frown)
        'brow_lowerer_right': 296,
        'eye_left_upper': 386,     # AU5: Upper lid raise
        'eye_right_upper': 159,
        'eye_left_lower': 374,
        'eye_right_lower': 145,
        'cheek_left': 50,          # AU6: Cheek raiser (Duchenne smile)
        'cheek_right': 280,
        'nose_wrinkler': 4,        # AU9: Nose wrinkler (disgust)
        'lip_corner_left': 61,     # AU12: Lip corner puller (smile)
        'lip_corner_right': 291,
        'lip_upper_inner': 13,     # AU20: Lip stretcher (fear)
        'lip_lower_inner': 14,
        'jaw': 152,                # AU26: Jaw drop (surprise)
    }
    
    def __init__(self, data_queue, stop_event, show_preview=True):
        super().__init__(daemon=True)
        self.queue = data_queue
        self.stop_event = stop_event
        self.show_preview = show_preview
        
        # Initialize MediaPipe face mesh
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enables iris tracking
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    
    def extract_features(self, landmarks, frame_shape):
        """
        Convert raw landmark positions to emotion-relevant features.
        Returns a flat dictionary of scalar values.
        """
        h, w = frame_shape[:2]
        features = {}
        
        # Get absolute positions for key landmarks
        lm_pos = {}
        for name, idx in self.LANDMARKS.items():
            lm = landmarks[idx]
            lm_pos[name] = np.array([lm.x * w, lm.y * h, lm.z * w])
        
        # ---- Brow Features ----
        # Brow height (relative to eye)
        features['brow_raise_left'] = (lm_pos['eye_left_upper'][1] - 
                                        lm_pos['brow_inner_left'][1]) / h
        features['brow_raise_right'] = (lm_pos['eye_right_upper'][1] - 
                                         lm_pos['brow_inner_right'][1]) / h
        features['brow_furrow'] = np.linalg.norm(
            lm_pos['brow_lowerer_left'] - lm_pos['brow_lowerer_right']) / w
        
        # ---- Eye Features ----
        # Eye openness
        features['eye_open_left'] = (lm_pos['eye_left_lower'][1] - 
                                      lm_pos['eye_left_upper'][1]) / h
        features['eye_open_right'] = (lm_pos['eye_right_lower'][1] - 
                                       lm_pos['eye_right_upper'][1]) / h
        
        # Iris position (if refine_landmarks enabled) - tracks gaze
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        features['gaze_x'] = (left_iris.x + right_iris.x) / 2 - 0.5
        features['gaze_y'] = (left_iris.y + right_iris.y) / 2 - 0.5
        
        # ---- Mouth Features ----
        features['mouth_open'] = (lm_pos['lip_lower_inner'][1] - 
                                   lm_pos['lip_upper_inner'][1]) / h
        mouth_width = np.linalg.norm(
            lm_pos['lip_corner_left'] - lm_pos['lip_corner_right']) / w
        features['mouth_width'] = mouth_width
        
        # Smile index: wide mouth + cheek raise
        features['smile_left'] = lm_pos['cheek_left'][1] / h
        features['smile_right'] = lm_pos['cheek_right'][1] / h
        
        # Jaw drop
        features['jaw_drop'] = lm_pos['jaw'][1] / h
        
        # ---- Head Pose ----
        # Nose tip as reference for head orientation
        nose_tip = landmarks[4]
        features['head_x'] = nose_tip.x - 0.5  # Horizontal rotation
        features['head_y'] = nose_tip.y - 0.5  # Vertical rotation
        
        return features
    
    def run(self):
        cap = cv2.VideoCapture(0)  # 0 = default webcam
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print("  WARNING: Could not open webcam.")
            return
        
        print("  Face camera active.")
        frame_count = 0
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            record = {
                'system_time': time.time(),
                'sensor': 'face',
                'frame': frame_count,
                'face_detected': False
            }
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features = self.extract_features(landmarks, frame.shape)
                record['face_detected'] = True
                record.update(features)
                
                if self.show_preview:
                    self.mp_draw.draw_landmarks(
                        frame,
                        results.multi_face_landmarks[0],
                        self.mp_face.FACEMESH_CONTOURS
                    )
            
            self.queue.put(record)
            
            if self.show_preview:
                cv2.imshow('Face Tracking Preview (press Q to hide)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.show_preview = False
                    cv2.destroyAllWindows()
        
        cap.release()
        cv2.destroyAllWindows()


# ============================================================
# SELF-REPORT: Triggered EMA Prompts
# ============================================================

class SelfReportCollector:
    """
    Prompts participants for self-report ratings at set intervals.
    Uses the SAM scale: Valence (1-9), Arousal (1-9), Dominance (1-9)
    """
    
    def __init__(self, data_queue, interval_seconds=120):
        self.queue = data_queue
        self.interval = interval_seconds
        self.ratings = []
    
    def collect_rating(self, trigger="scheduled"):
        print("\n" + "="*50)
        print("SELF-REPORT CHECK-IN")
        print("="*50)
        print("Rate your CURRENT emotional state:")
        print()
        print("VALENCE (how positive/negative do you feel?)")
        print("  1=Very negative  5=Neutral  9=Very positive")
        valence = self._get_rating("Valence (1-9): ", 1, 9)
        
        print("\nAROUSAL (how activated/calm do you feel?)")
        print("  1=Very calm      5=Moderate  9=Very activated")
        arousal = self._get_rating("Arousal (1-9): ", 1, 9)
        
        print("\nDOMINANCE (how in control do you feel?)")
        print("  1=No control     5=Moderate  9=Full control")
        dominance = self._get_rating("Dominance (1-9): ", 1, 9)
        
        print("\nIn one word, describe your main emotion (e.g., anxious, calm, excited):")
        label = input("Emotion word: ").strip().lower()
        
        record = {
            'system_time': time.time(),
            'sensor': 'self_report',
            'trigger': trigger,
            'valence': valence,
            'arousal': arousal,
            'dominance': dominance,
            'emotion_label': label
        }
        
        self.queue.put(record)
        self.ratings.append(record)
        print(f"\nRecorded. V={valence} A={arousal} D={dominance} [{label}]")
        print("="*50 + "\n")
        return record
    
    def _get_rating(self, prompt, min_val, max_val):
        while True:
            try:
                val = int(input(prompt))
                if min_val <= val <= max_val:
                    return val
                print(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a number.")


# ============================================================
# DATA WRITER: Saves everything to disk
# ============================================================

class DataWriter:
    """Consumes from the shared queue and writes data to CSV files."""
    
    def __init__(self, participant_id, session_num, condition, data_queue, stop_event):
        self.participant_id = participant_id
        self.session = session_num
        self.condition = condition
        self.queue = data_queue
        self.stop_event = stop_event
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = os.path.join(
            DATA_FOLDER, 
            f"{participant_id}_session{session_num}_{condition}_{timestamp}"
        )
        os.makedirs(self.folder, exist_ok=True)
        
        # Open separate files for each sensor
        self.files = {}
        self.writers = {}
        self.buffer = {'gsr': [], 'polar_h10': [], 'face': [], 'self_report': [], 'vr_events': []}
        
        self.metadata = {
            'participant_id': participant_id,
            'session': session_num,
            'condition': condition,
            'start_time': time.time(),
            'start_datetime': datetime.now().isoformat()
        }
        
        # Save metadata
        with open(os.path.join(self.folder, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n  Saving data to: {self.folder}")
    
    def run(self):
        """Continuously drain queue and buffer records."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                record = self.queue.get(timeout=0.1)
                sensor = record.get('sensor', 'unknown')
                if sensor in self.buffer:
                    self.buffer[sensor].append(record)
                
                # Flush buffer every 100 records
                if len(self.buffer[sensor]) >= 100:
                    self._flush(sensor)
                    
            except queue.Empty:
                continue
        
        # Flush all remaining data
        for sensor in self.buffer:
            self._flush(sensor)
        
        print(f"\n  Data saved to: {self.folder}")
    
    def _flush(self, sensor):
        if not self.buffer[sensor]:
            return
        df = pd.DataFrame(self.buffer[sensor])
        filepath = os.path.join(self.folder, f"{sensor}.csv")
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
        self.buffer[sensor] = []
    
    def log_vr_event(self, event_type, scene_name, details=None):
        """Call this when VR scenes change — critical for labeling."""
        record = {
            'system_time': time.time(),
            'sensor': 'vr_events',
            'event_type': event_type,
            'scene_name': scene_name,
            'details': json.dumps(details or {})
        }
        self.queue.put(record)


# ============================================================
# MAIN COLLECTION SESSION
# ============================================================

def run_session(participant_id, session_num, condition, duration_minutes=45):
    """
    Main function that runs a full data collection session.
    
    Args:
        participant_id: e.g., "P001"
        session_num: 1, 2, 3...
        condition: "BASELINE", "ANXIETY_EXPOSURE", "CRAVING_EXPOSURE", etc.
        duration_minutes: How long to collect data
    """
    
    print("\n" + "="*60)
    print("  EMOTION AI-VR DATA COLLECTION")
    print(f"  Participant: {participant_id} | Session: {session_num}")
    print(f"  Condition: {condition}")
    print(f"  Duration: {duration_minutes} minutes")
    print("="*60)
    
    # Shared resources
    data_queue = queue.Queue(maxsize=10000)
    stop_event = threading.Event()
    
    # Initialize sensors
    print("\nInitializing sensors...")
    
    gsr = GSRCollector(data_queue, stop_event)
    face = FaceCollector(data_queue, stop_event, show_preview=True)
    self_report = SelfReportCollector(data_queue, interval_seconds=300)  # Every 5 min
    
    try:
        gsr.connect()
    except Exception as e:
        print(f"  GSR connection failed: {e}. Continuing without GSR.")
    
    # Data writer runs in its own thread
    writer = DataWriter(participant_id, session_num, condition, data_queue, stop_event)
    writer_thread = threading.Thread(target=writer.run, daemon=False)
    
    print("\nAll sensors ready. Starting session...")
    print("Press CTRL+C to end session early.\n")
    
    # Start everything
    gsr.start()
    face.start()
    writer_thread.start()
    
    # BASELINE: First 2 minutes, participant just sits quietly
    print(">>> BASELINE PERIOD (2 minutes — participant sits quietly)")
    writer.log_vr_event("scene_start", "BASELINE_NEUTRAL")
    
    # Collect a self-report at start
    self_report.collect_rating(trigger="session_start")
    
    session_start = time.time()
    last_report_time = session_start
    REPORT_INTERVAL = 300  # seconds
    
    try:
        while True:
            elapsed = time.time() - session_start
            
            # Scheduled self-reports
            if time.time() - last_report_time >= REPORT_INTERVAL:
                self_report.collect_rating(trigger="scheduled")
                last_report_time = time.time()
            
            # Check if session is done
            if elapsed >= duration_minutes * 60:
                print(f"\nSession complete ({duration_minutes} minutes).")
                break
            
            # Print progress every 30 seconds
            if int(elapsed) % 30 == 0:
                q_size = data_queue.qsize()
                print(f"  [{int(elapsed//60):02d}:{int(elapsed%60):02d}] "
                      f"Queue: {q_size} records | "
                      f"GSR buffers: {len(writer.buffer['gsr'])} | "
                      f"Face buffers: {len(writer.buffer['face'])}")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    
    # Collect final self-report
    self_report.collect_rating(trigger="session_end")
    writer.log_vr_event("session_end", condition)
    
    # Shut down
    print("\nStopping sensors...")
    stop_event.set()
    gsr.join(timeout=3)
    face.join(timeout=3)
    writer_thread.join(timeout=10)
    gsr.close()
    
    print(f"\n✓ Session complete. Data in: {writer.folder}")
    return writer.folder


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionAI-VR Data Collector")
    parser.add_argument("--participant", default="P001", help="Participant ID")
    parser.add_argument("--session", type=int, default=1, help="Session number")
    parser.add_argument("--condition", default="BASELINE", 
                        choices=["BASELINE", "ANXIETY_EXPOSURE", "CRAVING_EXPOSURE",
                                 "POSITIVE_AFFECT", "NEUTRAL", "SOCIAL_STRESS"],
                        help="Experimental condition")
    parser.add_argument("--duration", type=int, default=45, help="Duration in minutes")
    
    args = parser.parse_args()
    
    data_folder = run_session(
        participant_id=args.participant,
        session_num=args.session,
        condition=args.condition,
        duration_minutes=args.duration
    )
