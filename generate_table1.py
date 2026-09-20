"""
FILE: generate_table1.py
PURPOSE: Produce the complete Table 1 (feature-level statistical significance
         table) for the paper's Method/Results sections.

         CHANGED: now imports its statistics from stats_common.py and uses
         the SAME complete-case sample (listwise deletion across all 15
         features) as univariate_logistic_regression.py (Table 5) and
         tuned_classification_models.py (Table 6/7), instead of dropping
         NaNs one feature at a time. This guarantees Table 4 and Table 5
         report identical Cohen's d / p-values for every feature, and
         matches the N used for model training/testing.

USAGE:
    python generate_table1.py --data_dir data/processed --output_dir analysis_results --scale episodic
    python generate_table1.py --data_dir data/processed --output_dir analysis_results --scale episodic --emotion_filter fear
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
import warnings
warnings.filterwarnings('ignore')

from stats_common import (
    cohens_d_pooled, welch_ttest, r_pb_exact, effect_size_zone,
    complete_case_sample, BONFERRONI_N, BONFERRONI_ALPHA,
)

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------
AROUSAL_THRESHOLD = 5.0

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]


# -----------------------------------------------------------------------
# DATA LOADING -- unchanged
# -----------------------------------------------------------------------

def load_features(data_dir, scale='episodic', emotion_filter=None,
                   max_per_session=100):
    pattern = os.path.join(data_dir, f'*/features_{scale}.csv')
    files = glob(pattern)
    if not files:
        files = glob(os.path.join(data_dir, f'**/features_{scale}.csv'),
                      recursive=True)

    print(f"Found {len(files)} {scale} feature files")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if df.empty or 'arousal' not in df.columns:
                continue

            folder = os.path.basename(os.path.dirname(f))
            df['session'] = folder

            emotion = 'unknown'
            for key in ['anger', 'fear', 'sadness', 'disgust',
                        'amusement', 'gratitude', 'neutral', 'tenderness']:
                if key in folder.lower():
                    emotion = key
                    break
            df['emotion'] = emotion

            if emotion_filter and emotion != emotion_filter:
                continue

            if len(df) > max_per_session:
                df = df.sample(n=max_per_session, random_state=42)

            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['arousal_binary'] = (combined['arousal'] >= AROUSAL_THRESHOLD
                                   ).astype(int)

    print(f"  Total windows: {len(combined)}")
    if emotion_filter:
        print(f"  Emotion filter: {emotion_filter}")
    high = combined['arousal_binary'].sum()
    print(f"  High arousal: {high} ({high/len(combined)*100:.1f}%)")
    print(f"  Low arousal:  {len(combined)-high} "
          f"({(len(combined)-high)/len(combined)*100:.1f}%)")
    return combined


# -----------------------------------------------------------------------
# TABLE 1 -- now built on the shared complete-case sample + Welch's t-test
# -----------------------------------------------------------------------

def generate_table1(df, label='ALL EMOTIONS'):
    clean, avail = complete_case_sample(df, REAL_SENSOR_FEATURES)
    n_dropped = len(df) - len(clean)

    high = clean[clean['arousal_binary'] == 1]
    low = clean[clean['arousal_binary'] == 0]

    print(f"\n{'='*90}")
    print(f"TABLE 1 -- {label}")
    print(f"Complete-case sample (listwise deletion across all {len(avail)} features): "
          f"N={len(clean)}  (dropped {n_dropped} of {len(df)} windows with "
          f">=1 missing feature -- same sample used in Table 5 and Table 6/7)")
    print(f"Threshold: arousal >= {AROUSAL_THRESHOLD}   High: n={len(high)}  Low: n={len(low)}")
    print(f"Bonferroni threshold: p < {BONFERRONI_ALPHA:.5f} (0.05 / {BONFERRONI_N})")
    print(f"{'='*90}")

    rows = []
    for feat in avail:
        h = high[feat].values
        l = low[feat].values
        if len(h) < 3 or len(l) < 3:
            continue

        t_stat, df_w, p_val = welch_ttest(h, l)
        d = cohens_d_pooled(h, l)
        n1, n2 = len(h), len(l)
        r = r_pb_exact(d, n1, n2) if not np.isnan(d) else np.nan

        rows.append({
            'Feature': feat,
            'High Mean (SD)': f"{h.mean():.3f} ({h.std():.3f})",
            'Low Mean (SD)': f"{l.mean():.3f} ({l.std():.3f})",
            't': round(t_stat, 3),
            'df': round(df_w, 1),
            'p': p_val,
            'p<.05': 'YES' if p_val < 0.05 else 'no',
            f'Bonferroni sig (p<{BONFERRONI_ALPHA:.4f})':
                'YES' if p_val < BONFERRONI_ALPHA else 'no',
            "Cohen's d": round(d, 3) if not np.isnan(d) else np.nan,
            'd zone': effect_size_zone(d, 'd') if not np.isnan(d) else '',
            'r_pb': round(r, 3) if not np.isnan(r) else np.nan,
            'r_pb zone': effect_size_zone(r, 'r') if not np.isnan(r) else '',
            'n_high': n1,
            'n_low': n2,
        })

    table = pd.DataFrame(rows)
    table['_abs_d'] = table["Cohen's d"].abs()
    table = table.sort_values('_abs_d', ascending=False).drop(columns='_abs_d')
    table = table.reset_index(drop=True)

    print(table.to_string(index=False))
    return table


def format_p_for_export(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--output_dir', default='analysis_results')
    parser.add_argument('--scale', default='episodic',
                         choices=['micro', 'momentary', 'episodic', 'session'])
    parser.add_argument('--emotion_filter', default=None,
                         help="e.g. 'fear' to restrict to Fear1 windows only")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label = f"{args.emotion_filter.upper()} ONLY" if args.emotion_filter else "ALL EMOTIONS"
    suffix = args.emotion_filter if args.emotion_filter else 'all_emotions'

    df = load_features(args.data_dir, scale=args.scale,
                        emotion_filter=args.emotion_filter)
    if df is None:
        print(f"ERROR: No processed {args.scale}-scale data found in {args.data_dir}")
        raise SystemExit(1)

    table1 = generate_table1(df, label=label)
    export = table1.copy()
    export['p'] = export['p'].apply(format_p_for_export)

    out_path = os.path.join(args.output_dir, f'table1_{suffix}.csv')
    export.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
