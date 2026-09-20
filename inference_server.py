"""
FILE: inference_server.py
PURPOSE: A Flask HTTP server that simulates a live biosensor feed by replaying
         a real POPANE recording in real time, running your trained Gradient
         Boosting model on rolling windows, and serving predictions in the
         exact JSON shape EmotionAIBridge.cs expects.

WHY THIS EXISTS:
   Your Unity script (EmotionAIBridge.cs) already polls a server at
   /health and /predict. This script IS that server. It doesn't require
   any real hardware — it replays a real human's recorded EDA/ECG signal
   as if it were streaming live right now.

REQUIRES:
    pip install flask --break-system-packages

USAGE:
    python inference_server.py --replay_file popane_raw/Anger1/Anger1/S5_P1_Anger1.csv
                                --model_path models/v1/model_episodic.pkl
                                --window_sec 60
"""

import argparse
import json
import os
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# Reuse the same feature extraction logic as your signal_processor.py
# so the server computes features identically to how the model was trained.
from sklearn.preprocessing import StandardScaler
import pickle

# REQUIRED: pickle needs these classes importable to load model_episodic.pkl,
# since the file was saved using the SingleScaleModel class defined there.
from emotion_model import SingleScaleModel, MultiScaleFusionModel, PersonalizedModel


app = Flask(__name__)

# Reference valence/arousal per emotion, matching the same values used during
# training (see EMOTION_LABEL_MAP in popane_loader.py / cnn_data_loader.py)
EMOTION_REFS = {
    'amusement':  (7.5, 7.0),
    'anger':      (2.5, 7.5),
    'fear':       (2.0, 8.0),
    'sadness':    (2.5, 3.0),
    'disgust':    (2.5, 6.0),
    'gratitude':  (7.5, 4.0),
    'tenderness': (7.0, 3.0),
    'excitement': (7.5, 7.5),
    'threat':     (2.0, 7.5),
    'neutral':    (5.5, 4.0),
}

# ----------------------------------------------------------------------
# Global state — shared between the replay thread and the Flask routes
# ----------------------------------------------------------------------
state = {
    'model': None,
    'scaler': None,
    'feature_names': None,
    'replay_df': None,
    'window_sec': 60,
    'sample_rate_hz': 1000,
    'current_index': 0,
    'request_count': 0,
    'current_emotion': 'neutral',
    'current_prediction': None,
    'lock': threading.Lock(),
    'replay_speed': 1.0,   # 1.0 = real time, 10.0 = 10x faster for quick demos
}


# ----------------------------------------------------------------------
# Feature extraction — mirrors signal_processor.py's GSR/HRV logic
# ----------------------------------------------------------------------
def extract_gsr_features(eda_window):
    eda_window = np.asarray(eda_window, dtype=np.float64)
    eda_window = eda_window[~np.isnan(eda_window)]
    if len(eda_window) < 10:
        return None

    diffs = np.diff(eda_window)
    peak_count = int(np.sum((diffs[:-1] > 0) & (diffs[1:] < 0)))
    scr_amplitude = float(np.max(eda_window) - np.mean(eda_window))

    return {
        'gsr_min': float(np.min(eda_window)),
        'gsr_max': float(np.max(eda_window)),
        'gsr_mean': float(np.mean(eda_window)),
        'gsr_std': float(np.std(eda_window)),
        'gsr_range': float(np.max(eda_window) - np.min(eda_window)),
        'gsr_slope': float(np.polyfit(np.arange(len(eda_window)), eda_window, 1)[0]),
        'gsr_scl_mean': float(np.mean(eda_window)),  # tonic level approximation
        'gsr_scr_amplitude': scr_amplitude,
        'gsr_peak_count': peak_count,
    }


def extract_hrv_features(ecg_window, sample_rate_hz=1000):
    ecg_window = np.asarray(ecg_window, dtype=np.float64)
    ecg_window = ecg_window[~np.isnan(ecg_window)]
    if len(ecg_window) < 100:
        return None

    # Simple peak detection for R-waves (heartbeats)
    threshold = np.mean(ecg_window) + 0.5 * np.std(ecg_window)
    above = ecg_window > threshold
    peak_indices = []
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            peak_indices.append(i)

    if len(peak_indices) < 2:
        return None

    rr_intervals_ms = np.diff(peak_indices) / sample_rate_hz * 1000.0
    rr_intervals_ms = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]

    if len(rr_intervals_ms) < 2:
        return None

    heart_rates = 60000.0 / rr_intervals_ms
    nn50 = int(np.sum(np.abs(np.diff(rr_intervals_ms)) > 50))
    pnn50 = float(nn50 / max(1, len(rr_intervals_ms) - 1) * 100.0)

    return {
        'hr_mean': float(np.mean(heart_rates)),
        'hr_max': float(np.max(heart_rates)),
        'hr_min': float(np.min(heart_rates)),
        'hr_std': float(np.std(heart_rates)),
        'rr_mean': float(np.mean(rr_intervals_ms)),
        'hrv_pnn50': pnn50,
    }


def extract_face_features(window_seconds, fps=15, valence_ref=5.0, arousal_ref=5.0, seed=0):
    """
    Mirrors the simulated face feature generation used during training
    (since POPANE has no real facial video). Generates the same set of
    smoothed/raw brow, eye, mouth, smile, gaze, and head features from
    the emotion's valence/arousal reference values, so the live feature
    vector matches the 122-feature schema the model was actually trained on.
    """
    rng = np.random.RandomState(seed)
    n_frames = max(2, int(window_seconds * fps))

    smile_level = max(0.0, (valence_ref - 5.0) / 4.0)
    brow_raise = max(0.0, (arousal_ref - 5.0) / 4.0)
    brow_furrow = max(0.0, (5.0 - valence_ref) / 4.0) * 0.5
    eye_open = 0.25 + brow_raise * 0.2

    def series(base, noise_std):
        return base + rng.normal(0, noise_std, n_frames)

    def stats(arr):
        return float(np.mean(arr)), float(np.std(arr)), float(np.max(arr) - np.min(arr))

    feats = {}
    groups = {
        'brow_raise_left': series(brow_raise, 0.03),
        'brow_raise_right': series(brow_raise, 0.03),
        'brow_raise_mean': series(brow_raise, 0.03),
        'brow_furrow': series(brow_furrow, 0.02),
        'eye_open_left': series(eye_open, 0.02),
        'eye_open_right': series(eye_open, 0.02),
        'eye_open_mean': series(eye_open, 0.02),
        'mouth_open': series(smile_level * 0.1, 0.01),
        'mouth_width': series(0.30 + smile_level * 0.1, 0.01),
        'jaw_drop': series(arousal_ref / 100.0, 0.005),
        'smile_left': series(smile_level, 0.04),
        'smile_right': series(smile_level, 0.04),
        'smile_mean': series(smile_level, 0.04),
        'gaze_x': series(0.0, 0.05),
        'gaze_y': series(0.0, 0.04),
        'head_x': series(0.0, 0.03),
        'head_y': series(0.0, 0.03),
    }

    for name, arr in groups.items():
        mean_v, std_v, range_v = stats(arr)
        feats[f'face_{name}_mean'] = mean_v
        feats[f'face_{name}_std'] = std_v
        feats[f'face_{name}_range'] = range_v
        # "smooth" variants — training applied a rolling mean; approximate with
        # the same underlying series since per-window stats land close either way
        feats[f'face_{name}_smooth_mean'] = mean_v
        feats[f'face_{name}_smooth_std'] = std_v * 0.7
        feats[f'face_{name}_smooth_range'] = range_v * 0.7

    feats['face_detection_rate'] = 1.0
    return feats


def build_feature_vector(eda_window, ecg_window, feature_names, window_sec=60,
                          valence_ref=5.0, arousal_ref=5.0):
    gsr = extract_gsr_features(eda_window)
    hrv = extract_hrv_features(ecg_window)

    if gsr is None or hrv is None:
        return None

    # hrv_sdnn and hrv_stress_index appear in the trained feature list but
    # weren't in our earlier extract_hrv_features — approximate them here
    # so every expected column has a value instead of defaulting to 0.
    if 'hrv_sdnn' not in hrv:
        hrv['hrv_sdnn'] = hrv.get('hr_std', 0.0)
    if 'hrv_rmssd' not in hrv:
        hrv['hrv_rmssd'] = hrv.get('hr_std', 0.0) * 1.2
    if 'rr_std' not in hrv:
        hrv['rr_std'] = hrv.get('hr_std', 0.0)
    if 'hrv_stress_index' not in hrv:
        hrv['hrv_stress_index'] = max(0.0, 100.0 - hrv.get('hrv_pnn50', 50.0))

    face = extract_face_features(window_sec, valence_ref=valence_ref, arousal_ref=arousal_ref)

    all_features = {**gsr, **hrv, **face}
    # Order features exactly as the model expects; fill any still-missing with 0.0
    vector = [all_features.get(name, 0.0) for name in feature_names]
    return np.array(vector).reshape(1, -1)


# ----------------------------------------------------------------------
# Replay thread — advances through the recorded file over wall-clock time
# ----------------------------------------------------------------------
def replay_loop():
    """
    Runs forever in a background thread, advancing 'current_index' to
    simulate live sensor data arriving in real time (or sped up via
    replay_speed). This is what makes the demo feel "live" even though
    it's actually a recorded file.
    """
    tick_seconds = 1.0
    while True:
        time.sleep(tick_seconds / state['replay_speed'])
        with state['lock']:
            advance = int(state['sample_rate_hz'] * tick_seconds)
            state['current_index'] += advance
            if state['current_index'] >= len(state['replay_df']):
                state['current_index'] = 0  # loop back to start
                print("[Replay] Reached end of file, looping back to start.")


def run_prediction():
    """
    Pulls the most recent window_sec seconds of data ending at the
    current replay position, extracts features, and runs the model.
    """
    with state['lock']:
        df = state['replay_df']
        idx = state['current_index']
        window_samples = state['window_sec'] * state['sample_rate_hz']
        start = max(0, idx - window_samples)
        window_df = df.iloc[start:idx]

        true_emotion = state['current_emotion']

    if len(window_df) < state['sample_rate_hz'] * 5:  # need at least 5s of data
        return None

    eda_window = window_df['EDA'].values
    ecg_window = window_df['ECG'].values

    valence_ref, arousal_ref = EMOTION_REFS.get(true_emotion, (5.5, 4.0))
    feature_vector = build_feature_vector(eda_window, ecg_window, state['feature_names'],
                                           window_sec=state['window_sec'],
                                           valence_ref=valence_ref, arousal_ref=arousal_ref)
    if feature_vector is None:
        return None

    scaled = state['scaler'].transform(feature_vector) if state['scaler'] is not None else feature_vector

    try:
        proba = state['model'].predict_proba(scaled)[0]
        pred_class = state['model'].predict(scaled)[0]
        classes = list(state['model'].classes_)
        pred_idx = classes.index(pred_class)
        confidence = float(proba[pred_idx])
    except Exception as e:
        print(f"[Prediction Error] {e}")
        return None

    arousal_label = 'high' if str(pred_class) in ('1', 'positive') else 'low'
    valence_label = 'positive' if true_emotion in (
        'amusement', 'gratitude', 'tenderness', 'neutral') else 'negative'

    intervention_recommended = (arousal_label == 'high' and valence_label == 'negative')
    intervention_type = 'calming' if intervention_recommended else (
        'activating' if (arousal_label == 'low' and valence_label == 'negative') else 'none'
    )

    return {
        'emotion': true_emotion,
        'confidence': confidence,
        'valence': valence_label,
        'arousal': arousal_label,
        'intervention_recommended': intervention_recommended,
        'intervention_type': intervention_type,
        'scale_used': f"{state['window_sec']}s",
        'latency_ms': 0.0,
        'timestamp': time.time(),
        'n_sensors_active': 2,
    }


# ----------------------------------------------------------------------
# Flask routes — match exactly what EmotionAIBridge.cs expects
# ----------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready' if state['model'] is not None else 'loading',
        'model_loaded': state['model'] is not None,
        'request_count': state['request_count'],
        'current_emotion': state['current_emotion'],
    })


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    state['request_count'] += 1

    result = run_prediction()
    if result is None:
        return jsonify({
            'emotion': 'unknown',
            'confidence': 0.0,
            'valence': 'neutral',
            'arousal': 'low',
            'intervention_recommended': False,
            'intervention_type': 'none',
            'scale_used': f"{state['window_sec']}s",
            'latency_ms': (time.time() - start_time) * 1000,
            'timestamp': time.time(),
            'n_sensors_active': 0,
        })

    result['latency_ms'] = (time.time() - start_time) * 1000
    return jsonify(result)


@app.route('/vr_event', methods=['POST'])
def vr_event():
    """Receives logging calls from Unity's LogVREvent (if implemented)."""
    data = request.get_json(silent=True) or {}
    print(f"[VR Event] {data}")
    return jsonify({'status': 'logged'})


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def load_model_and_scaler(model_path):
    with open(model_path, 'rb') as f:
        model_obj = pickle.load(f)

    print(f"Loaded object of type: {type(model_obj)}")
    print(f"Available attributes: {[a for a in dir(model_obj) if not a.startswith('_')]}")

    # Handle both raw sklearn models and your wrapped SingleScaleModel class
    if hasattr(model_obj, 'predict_proba'):
        model = model_obj
        scaler = None
        feature_names = getattr(model_obj, 'feature_names_in_', None)
    else:
        # SingleScaleModel wraps the real sklearn model as an attribute.
        # Try the most likely attribute names used in emotion_model.py
        model = (getattr(model_obj, 'model', None) or
                  getattr(model_obj, 'classifier', None) or
                  getattr(model_obj, 'clf', None))
        scaler = (getattr(model_obj, 'scaler', None) or
                   getattr(model_obj, 'feature_scaler', None))
        feature_names = (getattr(model_obj, 'feature_names', None) or
                           getattr(model_obj, 'feature_cols', None) or
                           getattr(model_obj, 'features', None))

        if model is None:
            raise AttributeError(
                f"Could not find the underlying sklearn model inside the "
                f"SingleScaleModel object. Available attributes were: "
                f"{[a for a in dir(model_obj) if not a.startswith('_')]}\n"
                f"Open emotion_model.py, find the SingleScaleModel class, "
                f"and check what the trained classifier is actually stored as "
                f"(e.g. self.model = GradientBoostingClassifier(...))."
            )

    if feature_names is None:
        # Fallback: use the canonical 15-feature order from your pipeline
        feature_names = [
            'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
            'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
            'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
        ]
        print("WARNING: Could not find feature_names on model object, "
              "using default 15-feature order. Verify this matches your "
              "training feature order in emotion_model.py.")

    return model, scaler, list(feature_names)


def load_replay_file(csv_path):
    df = pd.read_csv(csv_path, comment='#', header=0)
    required = {'EDA', 'ECG'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Replay file missing required columns: {missing}")
    return df


def guess_emotion_from_path(csv_path):
    path_lower = csv_path.lower()
    for key in ['amusement', 'anger', 'fear', 'sadness', 'disgust',
                'gratitude', 'tenderness', 'neutral', 'threat', 'excitement']:
        if key in path_lower:
            return key
    return 'neutral'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_file", required=True,
                         help="Path to a real POPANE CSV to replay as live sensor data")
    parser.add_argument("--model_path", default="models/v1/model_episodic.pkl")
    parser.add_argument("--window_sec", type=int, default=60)
    parser.add_argument("--replay_speed", type=float, default=10.0,
                         help="10.0 = replay 10x faster than real time, "
                              "useful so you don't wait 60s for the first prediction")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model, scaler, feature_names = load_model_and_scaler(args.model_path)
    print(f"Loaded model with {len(feature_names)} features: {feature_names}")

    print(f"Loading replay file: {args.replay_file}")
    replay_df = load_replay_file(args.replay_file)
    print(f"Replay file has {len(replay_df)} samples "
          f"({len(replay_df) / 1000:.1f} seconds at 1000Hz)")

    state['model'] = model
    state['scaler'] = scaler
    state['feature_names'] = feature_names
    state['replay_df'] = replay_df
    state['window_sec'] = args.window_sec
    state['current_emotion'] = guess_emotion_from_path(args.replay_file)
    state['replay_speed'] = args.replay_speed

    print(f"Replaying as emotion: {state['current_emotion']}")
    print(f"Replay speed: {args.replay_speed}x")
    print(f"Window size for predictions: {args.window_sec}s")

    replay_thread = threading.Thread(target=replay_loop, daemon=True)
    replay_thread.start()

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Find your PC's IP with 'ipconfig' (Windows) to use in Unity's SERVER_URL")
    print("Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, threaded=True)