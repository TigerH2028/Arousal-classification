"""
FILE: signal_processor.py
PURPOSE: Cleans raw sensor data and extracts features at multiple time scales.
         Run this AFTER a data collection session, before model training.

HOW TO RUN:
  python signal_processor.py --session_folder data/raw/P001_session1_BASELINE_20240101_120000

OUTPUT: Creates data/processed/[session_name]/ with cleaned feature files.
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import neurokit2 as nk
import os
import json
import argparse
from pathlib import Path


# ============================================================
# TIME SCALE WINDOWS FOR FEATURE EXTRACTION
# ============================================================

TIME_WINDOWS = {
    'micro':     2,     # 2 seconds — startle responses
    'momentary': 15,    # 15 seconds — quick emotional shifts
    'episodic':  60,    # 1 minute — sustained states
    'session':   300,   # 5 minutes — mood arcs
}

STEP_SIZES = {
    'micro':     0.5,   # Slide window every 0.5 seconds
    'momentary': 5.0,
    'episodic':  15.0,
    'session':   60.0,
}


# ============================================================
# GSR / EDA PROCESSING
# ============================================================

class GSRProcessor:
    """
    Processes raw Galvanic Skin Response data.
    
    GSR has two components we separate:
    - SCL (Skin Conductance Level): slow baseline — reflects sustained stress
    - SCR (Skin Conductance Response): fast peaks — reflects discrete events
    """
    
    def __init__(self, sample_rate=20):
        self.fs = sample_rate
    
    def process(self, df):
        """Full GSR processing pipeline."""
        if df.empty:
            return pd.DataFrame()
        
        print("  Processing GSR...")
        
        # Sort by time
        df = df.sort_values('system_time').reset_index(drop=True)
        signal_raw = df['conductance_us'].values
        
        # Step 1: Remove outliers (values > 3 SD from rolling mean)
        rolling_mean = pd.Series(signal_raw).rolling(self.fs * 5, center=True).mean()
        rolling_std = pd.Series(signal_raw).rolling(self.fs * 5, center=True).std()
        outlier_mask = np.abs(signal_raw - rolling_mean) > (3 * rolling_std)
        signal_clean = signal_raw.copy()
        signal_clean[outlier_mask] = np.interp(
            np.where(outlier_mask)[0],
            np.where(~outlier_mask)[0],
            signal_raw[~outlier_mask]
        )
        
        # Step 2: Lowpass filter (remove high-frequency noise)
        # GSR changes slowly — filter out anything above 5 Hz
        b, a = signal.butter(4, 5.0 / (self.fs / 2), btype='low')
        signal_filtered = signal.filtfilt(b, a, signal_clean)
        
        # Step 3: Separate SCL and SCR using NeuroKit2
        try:
            eda_processed = nk.eda_process(signal_filtered, sampling_rate=self.fs)
            eda_df = eda_processed[0]
            
            df['gsr_clean'] = signal_filtered
            df['gsr_scl'] = eda_df['EDA_Tonic'].values[:len(df)]     # Slow component
            df['gsr_scr'] = eda_df['EDA_Phasic'].values[:len(df)]    # Fast peaks
            df['gsr_peaks'] = eda_df['SCR_Peaks'].values[:len(df)]   # Peak locations (0/1)
        except Exception as e:
            print(f"    NeuroKit2 EDA processing failed: {e}. Using basic filter only.")
            df['gsr_clean'] = signal_filtered
            df['gsr_scl'] = signal_filtered  # Fallback: use filtered signal
            df['gsr_scr'] = np.zeros(len(df))
            df['gsr_peaks'] = np.zeros(len(df))
        
        # Step 4: Z-score normalize (make person-relative)
        df['gsr_normalized'] = (df['gsr_clean'] - df['gsr_clean'].mean()) / \
                                 (df['gsr_clean'].std() + 1e-8)
        
        return df
    
    def extract_features_window(self, window_df):
        """Extract features from a time window of GSR data."""
        if window_df.empty:
            return {}
        
        gsr = window_df['gsr_clean'].values if 'gsr_clean' in window_df else \
              window_df['conductance_us'].values
        
        features = {
            'gsr_mean': np.mean(gsr),
            'gsr_std': np.std(gsr),
            'gsr_min': np.min(gsr),
            'gsr_max': np.max(gsr),
            'gsr_range': np.max(gsr) - np.min(gsr),
            'gsr_slope': np.polyfit(np.arange(len(gsr)), gsr, 1)[0],  # Trend
            'gsr_peak_count': int(window_df.get('gsr_peaks', pd.Series([0])).sum()),
            'gsr_scl_mean': window_df.get('gsr_scl', pd.Series([0])).mean(),
            'gsr_scr_amplitude': window_df.get('gsr_scr', pd.Series([0])).max(),
        }
        return features


# ============================================================
# HRV PROCESSING
# ============================================================

class HRVProcessor:
    """
    Processes heart rate and RR interval data.
    
    HRV (Heart Rate Variability) is a powerful autonomic nervous system marker.
    Low HRV = stress/anxiety. High HRV = calm/regulated.
    """
    
    def process(self, df):
        """Process raw HR + RR interval data."""
        if df.empty:
            return pd.DataFrame()
        
        print("  Processing HRV...")
        
        df = df.sort_values('system_time').reset_index(drop=True)
        
        # Explode RR intervals (each row may have multiple RR values)
        rr_records = []
        for _, row in df.iterrows():
            rr_list = row.get('rr_intervals_ms', [])
            if isinstance(rr_list, str):
                try:
                    rr_list = json.loads(rr_list.replace("'", '"'))
                except:
                    try:
                        rr_list = [float(rr_list)]
                    except:
                        rr_list = []
            elif isinstance(rr_list, (int, float)):
                import math
                rr_list = [] if math.isnan(float(rr_list)) else [float(rr_list)]
            if rr_list:
                for rr in rr_list:
                    rr_records.append({
                        'system_time': row['system_time'],
                        'heart_rate': row.get('heart_rate_bpm', np.nan),
                        'rr_ms': float(rr)
                    })
        
        if not rr_records:
            return pd.DataFrame()
        
        rr_df = pd.DataFrame(rr_records)
        
        # Filter physiologically plausible RR intervals (300-2000ms = 30-200 BPM)
        rr_df = rr_df[(rr_df['rr_ms'] > 300) & (rr_df['rr_ms'] < 2000)]
        
        return rr_df
    
    def extract_features_window(self, window_rr):
        """
        Extract HRV features from a window of RR intervals.
        Returns time-domain and basic frequency-domain features.
        """
        if len(window_rr) < 4:
            return {}
        
        rr = window_rr['rr_ms'].values
        hr = window_rr['heart_rate'].values
        
        # Time-domain features
        successive_diffs = np.diff(rr)
        
        features = {
            # Heart rate
            'hr_mean': np.mean(hr),
            'hr_std': np.std(hr),
            'hr_min': np.min(hr),
            'hr_max': np.max(hr),
            
            # RR interval statistics
            'rr_mean': np.mean(rr),
            'rr_std': np.std(rr),
            
            # HRV time-domain metrics
            'hrv_sdnn': np.std(rr),                                    # SD of NN intervals
            'hrv_rmssd': np.sqrt(np.mean(successive_diffs**2)),        # Root mean square SD
            'hrv_pnn50': np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100,
            
            # Stress proxy: low RMSSD = high stress
            'hrv_stress_index': 1.0 / (np.sqrt(np.mean(successive_diffs**2)) + 1e-8),
        }
        
        # Frequency domain: LF/HF ratio (requires ≥60 second windows)
        if len(rr) >= 20 and window_rr['rr_ms'].count() >= 20:
            try:
                freq_features = nk.hrv_frequency(
                    pd.DataFrame({'RR_Intervals': rr}), 
                    sampling_rate=4  # Interpolated HRV signal
                )
                features['hrv_lf_hf_ratio'] = float(freq_features.get('HRV_LFHF', [np.nan])[0])
                features['hrv_hf_power'] = float(freq_features.get('HRV_HF', [np.nan])[0])
                features['hrv_lf_power'] = float(freq_features.get('HRV_LF', [np.nan])[0])
            except:
                features['hrv_lf_hf_ratio'] = np.nan
                features['hrv_hf_power'] = np.nan
                features['hrv_lf_power'] = np.nan
        
        return features


# ============================================================
# FACE PROCESSING
# ============================================================

class FaceProcessor:
    """
    Processes facial landmark data into emotion-relevant features.
    Maps landmark movements to Action Unit (AU) approximations.
    """
    
    EMOTION_PROTOTYPES = {
        # Simplified: {AU_signature} -> emotion
        # Based on Ekman's FACS system (approximate)
        'happiness':  {'brow_raise_mean': 0, 'mouth_width': 0.45, 'smile_mean': 0.7},
        'sadness':    {'brow_raise_mean': 0.3, 'brow_furrow': 0.2, 'mouth_open': 0.1},
        'anger':      {'brow_furrow': 0.4, 'brow_raise_mean': -0.1, 'eye_open_mean': 0.3},
        'fear':       {'brow_raise_mean': 0.5, 'eye_open_mean': 0.5, 'mouth_open': 0.3},
        'surprise':   {'brow_raise_mean': 0.6, 'eye_open_mean': 0.6, 'jaw_drop': 0.5},
        'disgust':    {'brow_furrow': 0.3, 'mouth_width': -0.1},
        'contempt':   {'mouth_width': 0.1},   # Asymmetric — hard without left/right sep
        'neutral':    {'brow_raise_mean': 0,  'mouth_open': 0, 'eye_open_mean': 0.2},
    }
    
    def process(self, df):
        """Clean and aggregate face landmark data."""
        if df.empty:
            return pd.DataFrame()
        
        print("  Processing face data...")
        
        df = df.sort_values('system_time').reset_index(drop=True)
        face_df = df[df['face_detected'] == True].copy()
        
        if face_df.empty:
            print("    WARNING: No frames with detected faces.")
            return pd.DataFrame()
        
        # Smooth all landmark features with rolling average (reduces jitter)
        feature_cols = [c for c in face_df.columns 
                        if c not in ['system_time', 'sensor', 'frame', 'face_detected']]
        
        for col in feature_cols:
            if face_df[col].dtype in [np.float64, np.float32, float]:
                face_df[f'{col}_smooth'] = face_df[col].rolling(5, center=True).mean()
        
        # Add composite features
        face_df['brow_raise_mean'] = (face_df['brow_raise_left'] + 
                                       face_df['brow_raise_right']) / 2
        face_df['eye_open_mean'] = (face_df['eye_open_left'] + 
                                     face_df['eye_open_right']) / 2
        face_df['smile_mean'] = (face_df['smile_left'] + face_df['smile_right']) / 2
        
        return face_df
    
    def extract_features_window(self, window_df):
        """Extract features from a time window of face data."""
        if window_df.empty:
            return {'face_detection_rate': 0.0}
        
        numeric_cols = window_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['frame', 'system_time']]
        
        features = {'face_detection_rate': len(window_df) / max(len(window_df), 1)}
        
        for col in numeric_cols:
            vals = window_df[col].dropna().values
            if len(vals) > 0:
                features[f'face_{col}_mean'] = np.mean(vals)
                features[f'face_{col}_std'] = np.std(vals)
                features[f'face_{col}_range'] = np.max(vals) - np.min(vals)
        
        return features


# ============================================================
# WINDOWED FEATURE EXTRACTION (Multi-Scale)
# ============================================================

class MultiScaleFeatureExtractor:
    """
    Applies sliding windows at each time scale to extract features.
    This is how you capture emotion at micro, momentary, episodic, and session scales.
    """
    
    def __init__(self):
        self.gsr_proc = GSRProcessor(sample_rate=20)
        self.hrv_proc = HRVProcessor()
        self.face_proc = FaceProcessor()
    
    def extract_all_scales(self, session_folder):
        """
        Main function: loads all sensor data from a session,
        processes it, and creates windowed feature matrices at each scale.
        """
        
        print(f"\nLoading session data from: {session_folder}")
        
        # Load raw data
        gsr_raw = self._load_csv(session_folder, 'gsr.csv')
        hr_raw = self._load_csv(session_folder, 'polar_h10.csv')
        face_raw = self._load_csv(session_folder, 'face.csv')
        events_raw = self._load_csv(session_folder, 'vr_events.csv')
        reports_raw = self._load_csv(session_folder, 'self_report.csv')
        
        # Process each sensor
        gsr_df = self.gsr_proc.process(gsr_raw)
        hr_df = self.hrv_proc.process(hr_raw)
        face_df = self.face_proc.process(face_raw)
        
        # Get time range
        all_times = []
        for df in [gsr_df, hr_df, face_df]:
            if not df.empty:
                all_times.extend(df['system_time'].values)
        
        if not all_times:
            print("ERROR: No valid sensor data found.")
            return {}
        
        t_start = min(all_times)
        t_end = max(all_times)
        
        # Create label mapping from VR events and self-reports
        labels = self._create_labels(events_raw, reports_raw, t_start, t_end)
        
        results = {}
        
        # Extract windowed features at each time scale
        for scale_name, window_sec in TIME_WINDOWS.items():
            step_sec = STEP_SIZES[scale_name]
            
            print(f"\nExtracting {scale_name} scale features "
                  f"(window={window_sec}s, step={step_sec}s)...")
            
            windows = []
            t = t_start + window_sec
            
            while t <= t_end:
                t_win_start = t - window_sec
                t_win_end = t
                
                # Extract window from each sensor
                gsr_win = gsr_df[(gsr_df['system_time'] >= t_win_start) & 
                                  (gsr_df['system_time'] < t_win_end)] if not gsr_df.empty else pd.DataFrame()
                hr_win = hr_df[(hr_df['system_time'] >= t_win_start) & 
                                (hr_df['system_time'] < t_win_end)] if not hr_df.empty else pd.DataFrame()
                face_win = face_df[(face_df['system_time'] >= t_win_start) & 
                                    (face_df['system_time'] < t_win_end)] if not face_df.empty else pd.DataFrame()
                
                # Extract features from each window
                feature_dict = {
                    'window_start': t_win_start - t_start,  # Relative to session start
                    'window_end': t_win_end - t_start,
                    'scale': scale_name,
                }
                
                feature_dict.update(self.gsr_proc.extract_features_window(gsr_win))
                feature_dict.update(self.hrv_proc.extract_features_window(hr_win))
                feature_dict.update(self.face_proc.extract_features_window(face_win))
                
                # Add label (what was the person feeling in this window?)
                label = self._get_label_for_window(labels, t_win_start, t_win_end)
                feature_dict.update(label)
                
                windows.append(feature_dict)
                t += step_sec
            
            scale_df = pd.DataFrame(windows)
            results[scale_name] = scale_df
            print(f"  Created {len(scale_df)} windows with {len(scale_df.columns)} features")
        
        # Save to disk
        output_dir = session_folder.replace('raw', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        for scale_name, df in results.items():
            path = os.path.join(output_dir, f"features_{scale_name}.csv")
            # Drop columns that are mostly empty before saving
            df = df.dropna(axis=1, thresh=max(1, int(len(df) * 0.05)))
            # Keep only numeric columns plus essential metadata
            meta_cols = ['window_start', 'window_end', 'scale', 'condition',
                 'valence', 'arousal', 'label_source']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            keep_cols = meta_cols + [c for c in numeric_cols if c not in meta_cols]
            keep_cols = [c for c in keep_cols if c in df.columns]
            df = df[keep_cols]
            df.to_csv(path, index=False)
            print(f"  Saved: {path} ({len(df)} rows x {len(df.columns)} cols)")
        
        # Save session-level summary
        summary = {
            'session_folder': session_folder,
            'n_windows': {k: len(v) for k, v in results.items()},
            'n_features': {k: len(v.columns) for k, v in results.items()},
            'duration_seconds': t_end - t_start,
            'sensors_available': {
                'gsr': not gsr_df.empty,
                'hrv': not hr_df.empty,
                'face': not face_df.empty,
            }
        }
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Feature extraction complete. Output in: {output_dir}")
        return results
    
    def _load_csv(self, folder, filename):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"  Loaded {filename}: {len(df)} rows")
                return df
            except Exception as e:
                print(f"  WARNING: Could not load {filename}: {e}")
        else:
            print(f"  No {filename} found (sensor not connected?)")
        return pd.DataFrame()
    
    def _create_labels(self, events_df, reports_df, t_start, t_end):
        """Create a timeline of known emotional states from VR events + self-reports."""
        labels = []
        
        # From VR events: what scene was active
        if not events_df.empty:
            for _, row in events_df.iterrows():
                labels.append({
                    'time': row['system_time'],
                    'source': 'vr_event',
                    'condition': row.get('scene_name', 'UNKNOWN'),
                    'valence': None,
                    'arousal': None,
                })
        
        # From self-reports: actual subjective ratings
        if not reports_df.empty:
            for _, row in reports_df.iterrows():
                labels.append({
                    'time': row['system_time'],
                    'source': 'self_report',
                    'condition': row.get('emotion_label', 'unknown'),
                    'valence': row.get('valence', None),
                    'arousal': row.get('arousal', None),
                })
        
        return sorted(labels, key=lambda x: x['time'])
    
    def _get_label_for_window(self, labels, t_start, t_end):
        """Find the most recent label that applies to this window."""
        applicable = [l for l in labels if l['time'] <= t_end]
        
        if not applicable:
            return {'condition': 'UNKNOWN', 'valence': np.nan, 'arousal': np.nan}
        
        most_recent = applicable[-1]
        
        # Also average any self-reports within the window
        in_window = [l for l in labels 
                     if t_start <= l['time'] <= t_end and l['source'] == 'self_report']
        
        if in_window:
            valences = [l['valence'] for l in in_window if l['valence'] is not None]
            arousals = [l['arousal'] for l in in_window if l['arousal'] is not None]
            return {
                'condition': most_recent['condition'],
                'valence': np.mean(valences) if valences else np.nan,
                'arousal': np.mean(arousals) if arousals else np.nan,
                'label_source': 'self_report'
            }
        
        return {
            'condition': most_recent['condition'],
            'valence': most_recent.get('valence', np.nan),
            'arousal': most_recent.get('arousal', np.nan),
            'label_source': 'vr_event'
        }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_folder", required=True, help="Path to raw session folder")
    args = parser.parse_args()
    
    extractor = MultiScaleFeatureExtractor()
    results = extractor.extract_all_scales(args.session_folder)
    
    print("\nFeature counts per scale:")
    for scale, df in results.items():
        print(f"  {scale}: {len(df)} windows × {len(df.columns)} features")
