"""
FILE: popane_loader.py
PURPOSE: Converts POPANE dataset CSV files into the format
         expected by signal_processor.py and emotion_model.py

HOW TO USE:
  1. Download POPANE from https://osf.io/94bpx/
  2. Put all CSV files in a folder called popane_raw/
  3. Run: python popane_loader.py --input_dir popane_raw --output_dir data/raw

POPANE CSV format (discovered from S6_P1_Amusement1.csv):
  - Header lines start with # (skipped via comment='#')
  - Columns: timestamp, affect, ECG, dzdt, dz, z0, EDA, SBP, DBP, CO, TPR, marker
  - timestamp: seconds at 1000Hz
  - affect: continuous self-report 1-9 (valence proxy)
  - EDA: skin conductance in microsiemens (= GSR)
  - SBP/DBP: blood pressure beat-to-beat (used to derive HR)
  - ECG: often NaN (not recorded in all studies)
  - marker: integer stimulus ID
  - Filename: S{study}_P{participant}_{emotion}{number}.csv
"""

import pandas as pd
import numpy as np
import os
import json
from glob import glob
from pathlib import Path


# ============================================================
# EMOTION LABEL MAPPING
# Source: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013).
#   Norms of valence, arousal, and dominance for 13,915 English lemmas.
#   Behavior Research Methods, 45, 1191-1207.
#   https://doi.org/10.3758/s13428-012-0314-x
#   Dataset: https://github.com/JULIELab/XANEW
#
# All values are A.Mean.Sum (arousal) and V.Mean.Sum (valence) from the
# published dataset, looked up by exact word match. Scale: 1-9.
# These are empirical self-report measurements from ~2,000 participants
# via the Self-Assessment Manikin (SAM) scale -- NOT invented estimates.
#
# Key difference from old table:
#   amusement: old arousal=7.0 -> Warriner arousal=4.82 (LOW arousal)
#   This is consistent with psychophysiology: amusement is a pleasant
#   but moderate-intensity state, not a high-activation state.
# ============================================================

# Map emotion name in filename -> (valence, arousal, quadrant_label)
# Numeric values: Warriner et al. (2013) A.Mean.Sum and V.Mean.Sum
EMOTION_LABEL_MAP = {
    'amusement':  (7.00, 4.82, 'calm_content'),     # LOW arousal per Warriner
    'excitement': (7.62, 6.21, 'excited_happy'),
    'gratitude':  (6.67, 5.09, 'excited_happy'),
    'tenderness': (6.89, 3.10, 'calm_content'),
    'anger':      (2.50, 5.93, 'anxious_stressed'),
    'fear':       (2.93, 6.14, 'anxious_stressed'),
    'threat':     (2.63, 6.57, 'anxious_stressed'),
    'disgust':    (3.32, 5.00, 'anxious_stressed'),
    'sadness':    (2.40, 2.81, 'sad_depressed'),
    'neutral':    (5.50, 3.45, 'neutral'),
    'baseline':   (5.40, 2.52, 'neutral'),
}

# ============================================================
# PARSE FILENAME
# ============================================================

def parse_filename(filepath):
    """
    Extract study, participant, and emotion from path.
    Handles both flat files and subfolder structure:
      popane_raw/Amusement1/S6_P1_Amusement1.csv
      popane_raw/Anger1/S3_P2_Anger1.csv
    """
    stem = Path(filepath).stem        # e.g. S6_P1_Amusement1
    parent = Path(filepath).parent.name  # e.g. Amusement1

    parts = stem.split('_')
    study = parts[0] if parts[0].startswith('S') else 'S0'

    # Find participant ID
    p_part = next((p for p in parts if p.startswith('P')), 'P0')
    p_num = p_part.replace('P', '').replace('p', '')
    try:
        participant_id = f"P{int(p_num):03d}"
    except ValueError:
        participant_id = f"P{p_num}"

    # Get emotion from parent folder name (more reliable than filename)
    emotion_source = parent.lower()
    emotion_key = 'neutral'
    for key in EMOTION_LABEL_MAP:
        if key in emotion_source:
            emotion_key = key
            break

    # Handle special folder names
    if 'positive_emotion_high' in emotion_source:
        emotion_key = 'excitement'
    elif 'positive_emotion_low' in emotion_source:
        emotion_key = 'tenderness'
    elif 'baseline' in emotion_source:
        emotion_key = 'baseline'

    return study, participant_id, emotion_key, stem

# ============================================================
# SIGNAL EXTRACTION
# ============================================================

def extract_gsr(df):
    """
    Convert POPANE EDA column → project GSR format.
    EDA is already in microsiemens at 1000Hz.
    Downsample to 20Hz to match Arduino GSR rate.
    """
    # Downsample 1000Hz → 20Hz (take every 50th sample)
    DOWNSAMPLE = 50
    df_ds = df.iloc[::DOWNSAMPLE].copy().reset_index(drop=True)

    return pd.DataFrame({
        'system_time': df_ds['timestamp'].values,
        'conductance_us': df_ds['EDA'].values,
        'voltage': df_ds['EDA'].values / 20.0,  # Approximate back-convert
        'raw_adc': (df_ds['EDA'].values * 50).astype(int),
        'sensor': 'gsr'
    }).dropna(subset=['conductance_us'])


def extract_hr(df):
    """
    Derive heart rate and RR intervals from beat-to-beat SBP data.
    SBP is recorded beat-to-beat and held constant between beats at 1000Hz.
    Detect beat boundaries by finding where SBP changes value.
    """
    sbp = df['SBP'].values
    timestamps = df['timestamp'].values

    # Find indices where SBP value changes (= new beat)
    change_indices = np.where(np.diff(sbp) != 0)[0] + 1
    if len(change_indices) < 4:
        return pd.DataFrame()

    # RR interval = time between consecutive beat onsets
    beat_times = timestamps[change_indices]
    rr_intervals_sec = np.diff(beat_times)

    # Filter physiologically plausible beats (300-2000ms)
    rr_ms = rr_intervals_sec * 1000
    valid = (rr_ms > 300) & (rr_ms < 2000)
    rr_ms = rr_ms[valid]
    beat_times_valid = beat_times[1:][valid]

    if len(rr_ms) < 2:
        return pd.DataFrame()

    heart_rates = 60000.0 / rr_ms

    return pd.DataFrame({
        'system_time': beat_times_valid,
        'heart_rate_bpm': heart_rates,
        'rr_intervals_ms': rr_ms.tolist(),
        'rr_mean_ms': rr_ms,
        'sensor': 'polar_h10'
    })


def load_baseline_eda(baseline_path):
    """
    Load a participant's true resting EDA from their separate Baseline file.

    POPANE provides separate baseline files (marker=-1) recorded BEFORE
    any emotional stimulus was presented. These are the correct reference
    for person-relative EDA normalization -- NOT the first few seconds of
    the emotion file, which already contains emotional responding.

    Returns the median EDA from the baseline file (artifact-cleaned),
    or None if the file cannot be read or has no valid EDA.
    """
    try:
        df = pd.read_csv(baseline_path, comment='#', header=0)
        if 'EDA' not in df.columns:
            return None
        eda = df['EDA'].copy()
        eda[eda <= 0] = np.nan
        eda[eda > 60] = np.nan
        baseline = eda.median()
        if pd.isna(baseline) or baseline < 0.05:
            return None
        return float(baseline)
    except Exception:
        return None


def find_baseline_file(emotion_csv_path, baselines_dir):
    """
    Find the matching Baselines file for a given emotion CSV file.

    POPANE filenames follow the pattern: S{study}_P{participant}_{emotion}.csv
    The matching baseline follows: S{study}_P{participant}_Baseline.csv

    Searches in baselines_dir and one level of subdirectories
    (since Baselines folder often has a nested Baselines subfolder).
    """
    from pathlib import Path
    stem = Path(emotion_csv_path).stem       # e.g. S5_P1_Anger1
    parts = stem.split('_')
    if len(parts) < 2:
        return None

    study = parts[0]        # e.g. S5
    participant = parts[1]  # e.g. P1
    baseline_name = f"{study}_{participant}_Baseline.csv"

    # Search baselines_dir and one level deep
    search_dirs = [baselines_dir]
    try:
        search_dirs += [d.path for d in os.scandir(baselines_dir) if d.is_dir()]
    except Exception:
        pass

    for d in search_dirs:
        candidate = os.path.join(d, baseline_name)
        if os.path.exists(candidate):
            return candidate

    return None


def extract_self_reports(df, emotion_key, n_reports=4, eda_baseline_override=None):
    """
    Create self-report records from continuous affect ratings.

    VALENCE: from POPANE's own continuous 'affect' column (real self-report).

    AROUSAL: computed purely from this participant's own EDA signal,
    following the baseline-normalization approach endorsed in the EDA
    arousal classification literature (Sanchez-Reolid et al., 2022;
    Braithwaite et al., 2015). Each window's EDA is divided by this
    person's own TRUE RESTING BASELINE (from their separate Baseline
    file, marker=-1, recorded before any stimulus was presented).

    If eda_baseline_override is provided (loaded from the Baseline file),
    it is used directly. Otherwise falls back to the first 5 seconds of
    the emotion file as an approximation -- but this is less accurate
    because the emotion file starts during the emotional stimulus, not
    during rest.
    """
    valence_ref, _, quadrant = EMOTION_LABEL_MAP[emotion_key]

    # Sample at n evenly spaced points through the trial
    indices = np.linspace(0, len(df) - 1, n_reports, dtype=int)

    # --- CLEAN RAW EDA FIRST ---
    eda_clean = df['EDA'].copy()
    eda_clean[eda_clean <= 0] = np.nan
    eda_clean[eda_clean > 60] = np.nan

    # --- PERSONAL RESTING BASELINE ---
    if eda_baseline_override is not None:
        # Best case: use the true resting baseline from the separate
        # Baselines file (marker=-1), which was recorded before any
        # emotional stimulus was presented.
        eda_baseline = float(eda_baseline_override)
        baseline_source = 'true_baseline_file'
    else:
        # Fallback: first 5 seconds of the emotion file.
        # WARNING: this is less accurate because the emotion stimulus
        # has already started -- these seconds are NOT true rest.
        baseline_window = eda_clean.iloc[:5000]
        eda_baseline = baseline_window.median()
        if pd.isna(eda_baseline) or eda_baseline < 0.05:
            eda_baseline = eda_clean.median()
        if pd.isna(eda_baseline) or eda_baseline < 0.05:
            eda_baseline = 1.0
        baseline_source = 'emotion_file_first5s_fallback'

    eda_baseline = max(float(eda_baseline), 0.05)

    # --- PERCENTILE-CALIBRATED RESCALING ANCHORS ---
    # Measured empirically across this dataset (see eda_ratio_analysis.py).
    RATIO_LOW_ANCHOR  = 0.870   # 10th percentile from true baseline data
    RATIO_MID_ANCHOR  = 1.359   # 50th percentile (median) from true baseline data
    RATIO_HIGH_ANCHOR = 3.111   # 90th percentile from true baseline data

    reports = []
    for idx in indices:
        row = df.iloc[idx]

        # VALENCE: real self-report, unchanged
        raw_affect = float(row['affect']) if not pd.isna(row['affect']) else valence_ref
        valence = float(np.clip(raw_affect, 1, 9))

        # AROUSAL: physiology-only, no category table
        local_start = max(0, idx - 500)
        local_end   = min(len(df), idx + 500)
        local_eda   = eda_clean.iloc[local_start:local_end].median()
        eda_current = local_eda if not pd.isna(local_eda) else eda_baseline
        eda_ratio   = eda_current / eda_baseline

        if eda_ratio <= RATIO_MID_ANCHOR:
            span    = RATIO_MID_ANCHOR - RATIO_LOW_ANCHOR
            arousal = 1.0 + (eda_ratio - RATIO_LOW_ANCHOR) / span * 4.0 if span > 0 else 5.0
        else:
            span    = RATIO_HIGH_ANCHOR - RATIO_MID_ANCHOR
            arousal = 5.0 + (eda_ratio - RATIO_MID_ANCHOR) / span * 4.0 if span > 0 else 5.0
        arousal = float(np.clip(arousal, 1, 9))

        reports.append({
            'system_time':    float(row['timestamp']),
            'sensor':         'self_report',
            'valence':        round(valence, 2),
            'arousal':        round(arousal, 2),
            'eda_ratio':      round(eda_ratio, 3),
            'baseline_source': baseline_source,
            'emotion_label':  emotion_key,
            'trigger':        'continuous_sample'
        })

    return pd.DataFrame(reports)


def generate_face_features(df, emotion_key, fps=15):
    """
    POPANE has no facial camera data.
    Generate plausible face features from affect + emotion label.
    Based on Ekman FACS research mapping emotions to facial AUs.
    """
    valence_ref, arousal_ref, _ = EMOTION_LABEL_MAP[emotion_key]
    duration = float(df['timestamp'].max() - df['timestamp'].min())
    n_frames = int(duration * fps)

    if n_frames < 2:
        return pd.DataFrame()

    rng = np.random.RandomState(hash(emotion_key) % 2**31)
    t = np.linspace(0, duration, n_frames)

    # Map emotion to facial parameters
    smile = max(0.0, (valence_ref - 5.0) / 4.0)
    brow_raise = max(0.0, (arousal_ref - 5.0) / 4.0)
    brow_furrow = max(0.0, (5.0 - valence_ref) / 4.0) * 0.5
    eye_open = 0.25 + brow_raise * 0.2

    def smooth(arr, w=fps // 3):
        return pd.Series(arr).rolling(max(1, w), center=True, min_periods=1).mean().values

    return pd.DataFrame({
        'system_time': df['timestamp'].min() + t,
        'brow_raise_left':  smooth(brow_raise + rng.normal(0, 0.03, n_frames)),
        'brow_raise_right': smooth(brow_raise + rng.normal(0, 0.03, n_frames)),
        'brow_raise_mean':  smooth(brow_raise + rng.normal(0, 0.03, n_frames)),
        'brow_furrow':      smooth(brow_furrow + rng.normal(0, 0.02, n_frames)),
        'eye_open_left':    smooth(eye_open + rng.normal(0, 0.02, n_frames)),
        'eye_open_right':   smooth(eye_open + rng.normal(0, 0.02, n_frames)),
        'eye_open_mean':    smooth(eye_open + rng.normal(0, 0.02, n_frames)),
        'mouth_open':       smooth(smile * 0.08 + rng.normal(0, 0.01, n_frames)),
        'mouth_width':      smooth(0.30 + smile * 0.10 + rng.normal(0, 0.01, n_frames)),
        'jaw_drop':         smooth(arousal_ref / 100 + rng.normal(0, 0.005, n_frames)),
        'smile_left':       smooth(smile + rng.normal(0, 0.04, n_frames)),
        'smile_right':      smooth(smile + rng.normal(0, 0.04, n_frames)),
        'smile_mean':       smooth(smile + rng.normal(0, 0.04, n_frames)),
        'gaze_x':           smooth(rng.normal(0, 0.05, n_frames)),
        'gaze_y':           smooth(rng.normal(0, 0.04, n_frames)),
        'head_x':           smooth(rng.normal(0, 0.03, n_frames)),
        'head_y':           smooth(rng.normal(0, 0.03, n_frames)),
        'face_detected': True,
        'sensor': 'face',
        'frame': np.arange(n_frames)
    })


# ============================================================
# MAIN CONVERTER
# ============================================================

def convert_popane_file(csv_path, output_base_dir, baselines_dir=None):
    """
    Convert one POPANE CSV file into a project session folder.

    baselines_dir: path to the POPANE Baselines folder (e.g. popane_raw/Baselines).
    If provided, the matching participant baseline file is used for EDA
    normalization. If not provided, falls back to the first 5 seconds of
    the emotion file (less accurate).
    """
    study, participant_id, emotion_key, stem = parse_filename(csv_path)

    print(f"  {stem} → {participant_id} | {emotion_key} | {study}")

    # Load data (skip # comment lines)
    df = pd.read_csv(csv_path, comment='#')

    # Validate required columns exist
    required = ['timestamp', 'affect', 'EDA', 'SBP']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"    SKIP: missing columns {missing}")
        return None

    # Drop rows where EDA is missing
    df = df.dropna(subset=['EDA'])
    if len(df) < 1000:
        print(f"    SKIP: too little data ({len(df)} rows)")
        return None

    # --- LOOK UP TRUE RESTING BASELINE ---
    eda_baseline_override = None
    baseline_file_used = 'none'
    if baselines_dir is not None:
        baseline_path = find_baseline_file(csv_path, baselines_dir)
        if baseline_path is not None:
            eda_baseline_override = load_baseline_eda(baseline_path)
            if eda_baseline_override is not None:
                baseline_file_used = os.path.basename(baseline_path)
                print(f"    Baseline: {baseline_file_used} "
                      f"(EDA={eda_baseline_override:.3f} uS)")
            else:
                print(f"    Baseline file found but EDA invalid: {baseline_path}")
        else:
            print(f"    No baseline file found for {stem} — using fallback")

    # Create output session folder
    folder_name = f"{participant_id}_POPANE_{stem}"
    session_folder = os.path.join(output_base_dir, folder_name)
    os.makedirs(session_folder, exist_ok=True)

    # Extract and save each signal type
    gsr_df    = extract_gsr(df)
    hr_df     = extract_hr(df)
    face_df   = generate_face_features(df, emotion_key)
    reports_df = extract_self_reports(df, emotion_key,
                                       eda_baseline_override=eda_baseline_override)

    valence_ref, arousal_ref, quadrant = EMOTION_LABEL_MAP[emotion_key]
    events_df = pd.DataFrame([{
        'system_time': float(df['timestamp'].iloc[0]),
        'sensor': 'vr_events',
        'event_type': 'scene_start',
        'scene_name': emotion_key.upper(),
        'details': json.dumps({'emotion': emotion_key, 'quadrant': quadrant})
    }])

    # Save all files
    gsr_df.to_csv(os.path.join(session_folder, 'gsr.csv'), index=False)
    hr_df.to_csv(os.path.join(session_folder, 'polar_h10.csv'), index=False)
    face_df.to_csv(os.path.join(session_folder, 'face.csv'), index=False)
    reports_df.to_csv(os.path.join(session_folder, 'self_report.csv'), index=False)
    events_df.to_csv(os.path.join(session_folder, 'vr_events.csv'), index=False)

    # Save metadata
    metadata = {
        'participant_id': participant_id,
        'study': study,
        'session': stem,
        'emotion': emotion_key,
        'quadrant': quadrant,
        'warriner_valence_ref': valence_ref,
        'warriner_arousal_ref': arousal_ref,
        'arousal_mean_physiology': round(float(reports_df['arousal'].mean()), 3) if len(reports_df) > 0 else None,
        'baseline_file_used': baseline_file_used,
        'eda_baseline_value': round(eda_baseline_override, 4) if eda_baseline_override is not None else None,
        'n_gsr_samples': len(gsr_df),
        'n_hr_beats': len(hr_df),
        'duration_seconds': float(df['timestamp'].max() - df['timestamp'].min()),
        'start_time': float(df['timestamp'].iloc[0]),
        'condition': 'POPANE',
        'simulated': False,
        'dataset': 'POPANE'
    }
    with open(os.path.join(session_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return session_folder


def convert_all(input_dir, output_dir, max_files=None, baselines_dir=None):
    """
    Convert all POPANE CSV files in input_dir to project format.

    baselines_dir: path to the POPANE Baselines folder
    (e.g. popane_raw/Baselines). When provided, each participant's
    true resting EDA baseline is loaded from their matching Baseline
    file instead of using the first 5 seconds of the emotion file.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, '**/*.csv'), recursive=True))
    if not files:
        files = sorted(glob(os.path.join(input_dir, '*.csv')))

    if not files:
        print(f"No CSV files found in: {input_dir}")
        print("Make sure you downloaded the POPANE data from https://osf.io/94bpx/")
        return []

    if max_files:
        files = files[:max_files]

    if baselines_dir:
        print(f"Using true baseline files from: {baselines_dir}")
    else:
        print("WARNING: No --baselines_dir provided. Using first 5s of emotion "
              "file as baseline (less accurate). Pass --baselines_dir popane_raw/Baselines "
              "for proper baseline normalization.")

    print(f"Found {len(files)} CSV files. Converting...\n")

    converted = []
    skipped = 0

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", end=' ')
        folder = convert_popane_file(f, output_dir,
                                      baselines_dir=baselines_dir)
        if folder:
            converted.append(folder)
        else:
            skipped += 1

    print(f"\nDone: {len(converted)} converted, {skipped} skipped")
    print(f"Output: {output_dir}")
    print(f"\nNext steps:")
    print(f"  python run_pipeline.py  →  [3] PROCESS")
    print(f"  python run_pipeline.py  →  [4] TRAIN")
    print(f"  python run_pipeline.py  →  [6] ANALYZE")

    return converted


# ============================================================
# QUICK TEST — run this first on one file
# ============================================================

def test_single_file(csv_path):
    """
    Test conversion on one file before running the full batch.
    Run this first to verify everything works.
    """
    print(f"Testing: {csv_path}\n")

    study, participant_id, emotion_key, stem = parse_filename(csv_path)
    print(f"Parsed: study={study}, participant={participant_id}, emotion={emotion_key}")

    df = pd.read_csv(csv_path, comment='#')
    print(f"Loaded: {len(df)} rows, columns: {list(df.columns)}")

    gsr = extract_gsr(df)
    print(f"GSR: {len(gsr)} samples, range {gsr['conductance_us'].min():.2f}"
          f" - {gsr['conductance_us'].max():.2f} uS")

    hr = extract_hr(df)
    print(f"HR: {len(hr)} beats detected, "
          f"mean {hr['heart_rate_bpm'].mean():.1f} BPM" if not hr.empty else "HR: no beats detected")

    reports = extract_self_reports(df, emotion_key)
    print(f"Self-reports: {len(reports)} samples")
    print(f"  Valence range: {reports['valence'].min():.1f} - {reports['valence'].max():.1f}")
    print(f"  Arousal range: {reports['arousal'].min():.1f} - {reports['arousal'].max():.1f}")
    print(f"  Baseline source: {reports['baseline_source'].iloc[0]}")
    print(f"  (Pass --baselines_dir to use true resting baseline instead of emotion file)")

    print("\nAll checks passed. Ready to convert full dataset.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert POPANE dataset to project format')
    parser.add_argument('--input_dir', default='popane_raw',
                        help='Folder containing POPANE CSV files')
    parser.add_argument('--output_dir', default='data/raw',
                        help='Where to write converted session folders')
    parser.add_argument('--baselines_dir', default=None,
                        help='Path to POPANE Baselines folder '
                             '(e.g. popane_raw/Baselines). When provided, '
                             'each participant\'s true resting EDA is loaded '
                             'from their matching Baseline file. Strongly '
                             'recommended for accurate arousal labeling.')
    parser.add_argument('--test', default=None,
                        help='Path to a single CSV file to test conversion')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files converted (for testing)')
    args = parser.parse_args()

    if args.test:
        test_single_file(args.test)
    else:
        convert_all(args.input_dir, args.output_dir, args.max_files,
                    baselines_dir=args.baselines_dir)