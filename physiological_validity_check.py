"""
FILE: physiological_validity_check.py
PURPOSE: Independent, model-free check of whether EDA and ECG signals
         actually differ between high-arousal and low-arousal recordings.

         This does NOT use Gradient Boosting or the CNN. It computes
         standard physiological summary metrics per raw POPANE file,
         splits files into "high arousal" vs "low arousal" groups based
         on emotion category, and reports descriptive statistics
         (mean, SD, IQR) plus box plots -- the same kind of validity
         check that should precede any predictive modeling.

METRICS COMPUTED (standard in psychophysiology literature):

  EDA / GSR:
    - scl_mean       Tonic Skin Conductance Level (baseline, microsiemens)
    - scr_amplitude  Largest phasic Skin Conductance Response in the window
    - scr_peak_count Number of distinct SCR events
    - eda_range      Max minus min conductance

  ECG / HRV:
    - hr_mean        Mean heart rate (BPM)
    - sdnn           SD of all RR intervals (global HRV, ms)
    - rmssd          Root mean square of successive RR differences (ms)
    - pnn50          % of consecutive RR intervals differing >50ms

USAGE:
    python physiological_validity_check.py --input_dir popane_raw --max_per_folder 25
"""

import numpy as np
import pandas as pd
import os
import argparse
import json
from pathlib import Path
from glob import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

SAMPLE_RATE_HZ = 1000

# Emotion category -> arousal group (matches your existing EMOTION_LABEL_MAP logic)
HIGH_AROUSAL_EMOTIONS = {'anger', 'fear', 'disgust', 'threat', 'excitement', 'amusement'}
LOW_AROUSAL_EMOTIONS = {'sadness', 'tenderness', 'gratitude', 'neutral', 'baseline'}


def classify_arousal_group(filepath):
    full_path_lower = str(filepath).lower()
    for key in HIGH_AROUSAL_EMOTIONS:
        if key in full_path_lower:
            return 'high', key
    for key in LOW_AROUSAL_EMOTIONS:
        if key in full_path_lower:
            return 'low', key
    return None, full_path_lower


def compute_eda_metrics(eda_signal):
    """Standard tonic/phasic EDA decomposition using a simple moving-average
    baseline (tonic) and residual peaks (phasic SCRs)."""
    eda_signal = eda_signal[~np.isnan(eda_signal)]
    if len(eda_signal) < SAMPLE_RATE_HZ * 2:
        return None

    # Tonic level: slow component via rolling mean (5-second window)
    window = SAMPLE_RATE_HZ * 5
    tonic = pd.Series(eda_signal).rolling(window, min_periods=1, center=True).mean().values
    phasic = eda_signal - tonic

    # SCR peaks: local maxima in the phasic component, minimum 1-second apart,
    # with a minimum prominence to avoid counting noise
    peaks, properties = find_peaks(phasic, distance=SAMPLE_RATE_HZ * 1,
                                    prominence=0.02)
    scr_amplitude = float(np.max(properties['prominences'])) if len(peaks) > 0 else 0.0

    return {
        'scl_mean': float(np.mean(tonic)),
        'scr_amplitude': scr_amplitude,
        'scr_peak_count': int(len(peaks)),
        'eda_range': float(np.max(eda_signal) - np.min(eda_signal)),
    }


def compute_ecg_metrics(ecg_signal):
    """Standard time-domain HRV metrics derived from R-peak detection."""
    ecg_signal = ecg_signal[~np.isnan(ecg_signal)]
    if len(ecg_signal) < SAMPLE_RATE_HZ * 2:
        return None

    # R-peak detection: minimum 300ms apart (corresponds to max 200 BPM)
    peaks, _ = find_peaks(ecg_signal, distance=int(SAMPLE_RATE_HZ * 0.3),
                           height=np.percentile(ecg_signal, 75))
    if len(peaks) < 3:
        return None

    rr_intervals_ms = np.diff(peaks) / SAMPLE_RATE_HZ * 1000
    rr_intervals_ms = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]
    if len(rr_intervals_ms) < 3:
        return None

    hr_bpm = 60000 / rr_intervals_ms
    successive_diffs = np.diff(rr_intervals_ms)

    sdnn = float(np.std(rr_intervals_ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2))) if len(successive_diffs) > 0 else np.nan
    pnn50 = float(np.mean(np.abs(successive_diffs) > 50) * 100) if len(successive_diffs) > 0 else np.nan

    return {
        'hr_mean': float(np.mean(hr_bpm)),
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
    }


def process_all_files(input_dir, max_per_folder=25):
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    rows = []

    for folder in subfolders:
        group, emotion = classify_arousal_group(folder)
        if group is None:
            print(f"  Skipping unclassified folder: {os.path.basename(folder)}")
            continue

        files = sorted(glob(os.path.join(folder, '**', '*.csv'), recursive=True))[:max_per_folder]
        print(f"  {os.path.basename(folder)} -> {group} arousal, {len(files)} files")

        for f in files:
            try:
                df = pd.read_csv(f, comment='#', header=0)
            except Exception as e:
                continue
            if not {'EDA', 'ECG'}.issubset(df.columns):
                continue

            eda_metrics = compute_eda_metrics(df['EDA'].values)
            ecg_metrics = compute_ecg_metrics(df['ECG'].values)
            if eda_metrics is None or ecg_metrics is None:
                continue

            row = {'file': os.path.basename(f), 'emotion': emotion, 'arousal_group': group}
            row.update(eda_metrics)
            row.update(ecg_metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def report_descriptive_stats(df, metrics, output_path='analysis_results/physiological_stats.txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("PHYSIOLOGICAL VALIDITY CHECK: HIGH vs LOW AROUSAL")
    lines.append("=" * 78)
    lines.append(f"\nTotal files analyzed: {len(df)}")
    lines.append(f"High arousal files: {(df['arousal_group']=='high').sum()}")
    lines.append(f"Low arousal files:  {(df['arousal_group']=='low').sum()}\n")

    from scipy import stats as scipy_stats

    for metric in metrics:
        high_vals = df.loc[df['arousal_group'] == 'high', metric].dropna()
        low_vals = df.loc[df['arousal_group'] == 'low', metric].dropna()

        if len(high_vals) < 2 or len(low_vals) < 2:
            continue

        t_stat, p_val = scipy_stats.ttest_ind(high_vals, low_vals, equal_var=False)

        lines.append("-" * 78)
        lines.append(f"METRIC: {metric}")
        lines.append("-" * 78)
        lines.append(f"  High arousal: mean={high_vals.mean():.3f}  SD={high_vals.std():.3f}  "
                      f"median={high_vals.median():.3f}  IQR=[{high_vals.quantile(0.25):.3f}, "
                      f"{high_vals.quantile(0.75):.3f}]")
        lines.append(f"  Low arousal:  mean={low_vals.mean():.3f}  SD={low_vals.std():.3f}  "
                      f"median={low_vals.median():.3f}  IQR=[{low_vals.quantile(0.25):.3f}, "
                      f"{low_vals.quantile(0.75):.3f}]")
        lines.append(f"  Difference in means: {high_vals.mean() - low_vals.mean():+.3f}")
        lines.append(f"  Largest single value (high arousal): {high_vals.max():.3f}")
        lines.append(f"  Largest single value (low arousal):  {low_vals.max():.3f}")
        lines.append(f"  Welch's t-test: t={t_stat:.3f}, p={p_val:.4f}"
                      f"  {'(significant, p<0.05)' if p_val < 0.05 else '(not significant)'}")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f"\nSaved full report to: {output_path}")


def plot_boxplots(df, metrics, output_path='analysis_results/physiological_boxplots.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n = len(metrics)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        high_vals = df.loc[df['arousal_group'] == 'high', metric].dropna()
        low_vals = df.loc[df['arousal_group'] == 'low', metric].dropna()

        bp = ax.boxplot([low_vals, high_vals], tick_labels=['Low', 'High'],
                         patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#4C72B0')
        bp['boxes'][1].set_facecolor('#C44E52')
        ax.set_title(metric)
        ax.set_ylabel('Value')

    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    fig.suptitle('EDA/ECG Physiological Metrics: High vs Low Arousal\n(IQR box plots, raw POPANE files, model-free)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved box plots: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                         help="Top-level popane_raw folder containing emotion subfolders")
    parser.add_argument("--max_per_folder", type=int, default=25)
    parser.add_argument("--output_dir", default="analysis_results")
    args = parser.parse_args()

    print("Scanning folders and computing physiological metrics...")
    df = process_all_files(args.input_dir, args.max_per_folder)

    if len(df) == 0:
        print("ERROR: No valid files processed. Check --input_dir path and folder names.")
    else:
        csv_path = os.path.join(args.output_dir, 'physiological_metrics_raw.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"\nSaved per-file metrics table: {csv_path}")

        metrics = ['scl_mean', 'scr_amplitude', 'scr_peak_count', 'eda_range',
                   'hr_mean', 'sdnn', 'rmssd', 'pnn50']

        report_descriptive_stats(df, metrics,
                                  output_path=os.path.join(args.output_dir, 'physiological_stats.txt'))
        plot_boxplots(df, metrics,
                       output_path=os.path.join(args.output_dir, 'physiological_boxplots.png'))