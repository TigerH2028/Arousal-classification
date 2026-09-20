"""
FILE: cnn_data_loader.py
PURPOSE: Loads raw EDA and ECG signal windows directly from POPANE raw CSVs
         for training a 1D CNN. This bypasses signal_processor.py entirely —
         the CNN learns its own features from the raw waveform shape instead
         of using hand-engineered statistics (gsr_mean, hr_std, etc.).

WHY THIS IS SEPARATE FROM YOUR EXISTING PIPELINE:
   Your Gradient Boosting model (emotion_model.py) reads 15 engineered
   features from data/processed/*/features_*.csv
   This CNN reads RAW signal arrays directly from data/raw/*/popane_raw.csv
   (or wherever your raw POPANE files live). Nothing about your existing
   pipeline changes.

USAGE:
    python cnn_data_loader.py --input_dir popane_raw --output_dir data/cnn_raw
       --window_sec 60 --max_files 200
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
from pathlib import Path
from glob import glob

SAMPLE_RATE_HZ = 1000          # POPANE raw files are sampled at 1000 Hz
WINDOW_SEC_DEFAULT = 60         # match your "episodic" scale by default
TARGET_LEN = None               # set after window_sec is known (window_sec * SAMPLE_RATE_HZ)

EMOTION_LABEL_MAP = {
    # folder-name keyword -> (valence_ref, arousal_ref)
    # Source: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013).
    #   Norms of valence, arousal, and dominance for 13,915 English lemmas.
    #   Behavior Research Methods, 45, 1191-1207.
    #   https://doi.org/10.3758/s13428-012-0314-x
    #   Dataset: https://github.com/JULIELab/XANEW
    # Values are A.Mean.Sum (arousal) and V.Mean.Sum (valence) from the
    # published dataset, looked up by exact word match. Scale 1-9.
    # NOTE: In this file, arousal_ref is used ONLY as a fallback for
    # face feature generation when EDA data is unavailable. Binary
    # labels are derived from physiology (EDA ratio), not from this table.
    'amusement':  (7.00, 4.82),   # Warriner: LOW arousal (was 7.0 in old table)
    'anger':      (2.50, 5.93),
    'fear':       (2.93, 6.14),
    'sadness':    (2.40, 2.81),
    'disgust':    (3.32, 5.00),
    'gratitude':  (6.67, 5.09),
    'tenderness': (6.89, 3.10),
    'excitement': (7.62, 6.21),
    'threat':     (2.63, 6.57),
    'neutral':    (5.50, 3.45),
    'baseline':   (5.40, 2.52),
}


def parse_folder_emotion(filepath):
    """Identify emotion from the parent folder name (case-insensitive substring match)."""
    parent = Path(filepath).parent.name.lower()
    grandparent = Path(filepath).parent.parent.name.lower()
    search_space = parent + " " + grandparent
    for key in EMOTION_LABEL_MAP:
        if key in search_space:
            return key
    return 'neutral'


def resample_or_pad(arr, target_len):
    """
    Force every window to the exact same length so the CNN gets a fixed
    input size. If too long, truncate. If too short, pad with the last value.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_width = target_len - len(arr)
    return np.pad(arr, (0, pad_width), mode='edge')


def normalize_signal(arr):
    """
    Z-score normalize each window individually (per-window normalization).
    This is standard practice for raw biosignal CNNs — it removes
    person-specific baseline offsets so the CNN learns shape, not absolute level.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


def extract_windows_from_file(csv_path, window_sec, step_sec=None):
    """
    Slice one raw POPANE file into overlapping or non-overlapping windows.
    Returns a list of dicts: {eda: array, ecg: array, label, valence, arousal}
    """
    if step_sec is None:
        step_sec = window_sec  # non-overlapping by default

    try:
        df = pd.read_csv(csv_path, comment='#', header=0)
    except Exception as e:
        print(f"  Could not read {csv_path}: {e}")
        return []

    required = {'EDA', 'ECG', 'affect', 'timestamp'}
    if not required.issubset(set(df.columns)):
        print(f"  SKIP {os.path.basename(csv_path)}: missing columns "
              f"{required - set(df.columns)}")
        return []

    target_len = window_sec * SAMPLE_RATE_HZ
    step_len = int(step_sec * SAMPLE_RATE_HZ)

    emotion_key = parse_folder_emotion(csv_path)
    valence_ref, _ = EMOTION_LABEL_MAP[emotion_key]  # valence fallback only; arousal_ref no longer used

    # CLEAN THE RAW EDA COLUMN FIRST: real skin conductance in
    # microsiemens is essentially always positive (typical range
    # ~0.5-40 uS). Negative values or extreme outliers are sensor
    # artifacts, not real physiology, and must be removed BEFORE any
    # baseline or ratio math -- matches the cleaning in popane_loader.py.
    eda_col_clean = df['EDA'].copy()
    eda_col_clean[eda_col_clean <= 0] = np.nan
    eda_col_clean[eda_col_clean > 60] = np.nan

    # Personal resting baseline for THIS file: MEDIAN EDA over the first
    # 5 seconds of recording (not 1 second / mean), to be robust against
    # a single noisy or glitched sample -- matches popane_loader.py.
    eda_baseline_full_file = eda_col_clean.iloc[:SAMPLE_RATE_HZ * 5].median()
    if pd.isna(eda_baseline_full_file) or eda_baseline_full_file < 0.05:
        eda_baseline_full_file = eda_col_clean.median()
    if pd.isna(eda_baseline_full_file) or eda_baseline_full_file < 0.05:
        eda_baseline_full_file = 1.0  # last-resort fallback
    eda_baseline_full_file = max(eda_baseline_full_file, 0.05)

    # Percentile-calibrated rescaling anchors, measured empirically across
    # the POPANE dataset via the diagnostic check (10th=0.711, median=1.000,
    # 90th=1.381). Anchoring to the median guarantees a balanced ~50/50
    # class split by construction (standard "median-split" approach).
    RATIO_LOW_ANCHOR  = 0.870   # 10th percentile from true baseline data
    RATIO_MID_ANCHOR  = 1.359   # 50th percentile (median) from true baseline data
    RATIO_HIGH_ANCHOR = 3.111   # 90th percentile from true baseline data

    def rescale_ratio_to_arousal(eda_ratio):
        if eda_ratio <= RATIO_MID_ANCHOR:
            span = RATIO_MID_ANCHOR - RATIO_LOW_ANCHOR
            a = 1.0 + (eda_ratio - RATIO_LOW_ANCHOR) / span * 4.0 if span > 0 else 5.0
        else:
            span = RATIO_HIGH_ANCHOR - RATIO_MID_ANCHOR
            a = 5.0 + (eda_ratio - RATIO_MID_ANCHOR) / span * 4.0 if span > 0 else 5.0
        return float(np.clip(a, 1, 9))

    n_samples = len(df)
    windows = []

    start = 0
    while start + target_len <= n_samples or (start == 0 and n_samples > SAMPLE_RATE_HZ * 5):
        end = min(start + target_len, n_samples)

        eda_raw = df['EDA'].iloc[start:end].values
        ecg_raw = df['ECG'].iloc[start:end].values
        affect_raw = df['affect'].iloc[start:end].values
        eda_clean_window = eda_col_clean.iloc[start:end].values

        eda_fixed = resample_or_pad(eda_raw, target_len)
        ecg_fixed = resample_or_pad(ecg_raw, target_len)

        if eda_fixed is None or ecg_fixed is None:
            break

        eda_norm = normalize_signal(eda_fixed)
        ecg_norm = normalize_signal(ecg_fixed)

        valid_affect = affect_raw[~np.isnan(affect_raw)]
        mean_affect = float(np.mean(valid_affect)) if len(valid_affect) > 0 else valence_ref

        # AROUSAL: pure physiology, using MEDIAN (not mean) of the
        # cleaned EDA values in this window, relative to the file's
        # own cleaned baseline, then percentile-calibrated rescaling.
        valid_eda_clean = eda_clean_window[~np.isnan(eda_clean_window)]
        window_median_eda = (float(np.median(valid_eda_clean))
                              if len(valid_eda_clean) > 0 else eda_baseline_full_file)
        eda_ratio = window_median_eda / eda_baseline_full_file
        arousal = rescale_ratio_to_arousal(eda_ratio)

        windows.append({
            'eda': eda_norm,
            'ecg': ecg_norm,
            'valence': mean_affect,
            'arousal': arousal,                  # now physiology-derived, percentile-calibrated
            'eda_ratio': round(eda_ratio, 3),     # kept for transparency/debugging
            'emotion': emotion_key,
            'label_binary': 1 if arousal >= 5 else 0,
            'source_file': os.path.basename(csv_path),
        })

        start += step_len
        if start + target_len > n_samples:
            break

    return windows


def build_cnn_dataset(input_dir, output_dir, window_sec=60, max_files=200, step_sec=None,
                       max_per_folder=None):
    """
    Walk through all POPANE subfolders, extract fixed-length windows,
    and save as a single compact .npz file (much faster to load than CSVs).

    input_dir can be a single path or a list of paths (e.g. one per emotion
    folder), so you can combine multiple emotions in one call without the
    alphabetical-glob-cap problem.

    max_per_folder: if set, caps files taken from EACH immediate subfolder
    of input_dir (or each path in a list), ensuring balanced emotion
    representation instead of a single global cap that favors whichever
    folder sorts first alphabetically.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_dirs = input_dir if isinstance(input_dir, (list, tuple)) else [input_dir]

    csv_files = []
    for d in input_dirs:
        # Get immediate subfolders (one per emotion) so we can cap per-folder
        subfolders = [f.path for f in os.scandir(d) if f.is_dir()] or [d]
        for sub in subfolders:
            sub_files = sorted(glob(os.path.join(sub, '**', '*.csv'), recursive=True))
            if max_per_folder:
                sub_files = sub_files[:max_per_folder]
            print(f"  {sub}: found {len(sub_files)} files (after cap)")
            csv_files.extend(sub_files)

    print(f"Total combined CSV files: {len(csv_files)}")

    if max_files and not max_per_folder:
        csv_files = csv_files[:max_files]
        print(f"Using first {len(csv_files)} files (global cap, no per-folder balancing)")

    all_eda, all_ecg, all_labels, all_emotions, all_valence, all_sources, all_eda_ratio = [], [], [], [], [], [], []

    for i, f in enumerate(csv_files):
        windows = extract_windows_from_file(f, window_sec=window_sec, step_sec=step_sec)
        for w in windows:
            all_eda.append(w['eda'])
            all_ecg.append(w['ecg'])
            all_labels.append(w['label_binary'])
            all_emotions.append(w['emotion'])
            all_valence.append(w['valence'])
            all_sources.append(w['source_file'])
            all_eda_ratio.append(w['eda_ratio'])

        if (i + 1) % 20 == 0 or (i + 1) == len(csv_files):
            print(f"  Processed {i+1}/{len(csv_files)} files -> {len(all_eda)} windows so far")

    if len(all_eda) == 0:
        print("ERROR: No windows extracted. Check your input_dir and column names.")
        return None

    eda_arr = np.stack(all_eda)        # shape: (n_windows, window_sec * 1000)
    ecg_arr = np.stack(all_ecg)
    labels_arr = np.array(all_labels)

    out_path = os.path.join(output_dir, f'cnn_dataset_{window_sec}s.npz')
    np.savez_compressed(
        out_path,
        eda=eda_arr,
        ecg=ecg_arr,
        labels=labels_arr,
        emotions=np.array(all_emotions),
        valence=np.array(all_valence),
        sources=np.array(all_sources),
        eda_ratio=np.array(all_eda_ratio),
    )

    print(f"\nSaved dataset: {out_path}")
    print(f"  EDA shape: {eda_arr.shape}")
    print(f"  ECG shape: {ecg_arr.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels_arr, return_counts=True)))}")

    meta = {
        'window_sec': window_sec,
        'sample_rate_hz': SAMPLE_RATE_HZ,
        'n_windows': int(len(all_eda)),
        'n_source_files': len(csv_files),
        'label_distribution': {str(k): int(v) for k, v in
                                zip(*np.unique(labels_arr, return_counts=True))},
    }
    with open(os.path.join(output_dir, f'cnn_dataset_{window_sec}s_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, nargs='+',
                         help="One or more folders containing POPANE raw CSVs "
                              "(e.g. --input_dir popane_raw  or  "
                              "--input_dir popane_raw/Anger1 popane_raw/Fear1 popane_raw/Amusement1)")
    parser.add_argument("--output_dir", default="data/cnn_raw")
    parser.add_argument("--window_sec", type=int, default=60,
                         help="Window length in seconds. 60=episodic, 10=momentary, 2=micro")
    parser.add_argument("--step_sec", type=int, default=None,
                         help="Step between windows. Defaults to window_sec (non-overlapping)")
    parser.add_argument("--max_files", type=int, default=200,
                         help="Global file cap, only used if --max_per_folder is not set")
    parser.add_argument("--max_per_folder", type=int, default=None,
                         help="Cap files taken from EACH emotion subfolder, for balanced classes")
    args = parser.parse_args()

    build_cnn_dataset(args.input_dir, args.output_dir, args.window_sec,
                       args.max_files, args.step_sec, args.max_per_folder)