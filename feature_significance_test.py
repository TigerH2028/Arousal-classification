"""
FILE: feature_significance_test.py
PURPOSE: Run Welch's t-test between high-arousal and low-arousal windows
         for EVERY numeric feature in the processed dataset (all 119 features,
         not just the 8 hand-computed physiological metrics from
         physiological_validity_check.py). Reports mean +/- SD per group,
         difference in means, t-statistic, p-value, and effect size (Cohen's d)
         for each feature, sorted by effect size. Produces a significance
         heatmap and a ranked bar chart.

WHY THIS MATTERS:
    The 15 features your models actually use (9 GSR + 6 HRV/HR) are embedded
    inside the full 119-feature processed files. This test tells you which
    of those features actually show a statistically meaningful difference
    between arousal groups -- before any model fitting, completely
    independently of model feature importance.

USAGE:
    python feature_significance_test.py --data_dir data/processed
                                         --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

AROUSAL_THRESHOLD = 5.0   # same threshold used in emotion_model.py

# The 15 real sensor features (everything else is synthetic face or metadata)
REAL_SENSOR_FEATURES = {
    'gsr': ['gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
            'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count'],
    'hrv': ['hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50'],
}
ALL_REAL_FEATURES = REAL_SENSOR_FEATURES['gsr'] + REAL_SENSOR_FEATURES['hrv']


def load_all_features(data_dir, scale='episodic', max_per_session=100):
    """Load all processed feature files for the given time scale."""
    pattern = os.path.join(data_dir, f'*/features_{scale}.csv')
    files = glob(pattern)
    if not files:
        pattern = os.path.join(data_dir, f'**/features_{scale}.csv')
        files = glob(pattern, recursive=True)

    print(f"Found {len(files)} {scale} feature files")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if df.empty or len(df.columns) == 0:
                continue
            if len(df) > max_per_session:
                df = df.sample(n=max_per_session, random_state=42)
            folder = os.path.basename(os.path.dirname(f))
            df['session'] = folder
            dfs.append(df)
        except Exception as e:
            continue

    if not dfs:
        print(f"ERROR: No valid {scale} feature files found in {data_dir}")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows loaded: {len(combined)}")
    return combined


def compute_cohens_d(group1, group2):
    """Cohen's d effect size: difference in means divided by pooled SD."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1-1)*group1.std()**2 + (n2-1)*group2.std()**2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def run_significance_tests(df, output_dir):
    """Run Welch's t-test for every numeric feature, grouped by arousal label."""
    os.makedirs(output_dir, exist_ok=True)

    if 'arousal' not in df.columns:
        print("ERROR: 'arousal' column not found. Check your processed files.")
        return None

    df['arousal_binary'] = (df['arousal'] >= AROUSAL_THRESHOLD).astype(int)
    high = df[df['arousal_binary'] == 1]
    low  = df[df['arousal_binary'] == 0]
    print(f"\nHigh arousal windows: {len(high)}")
    print(f"Low arousal windows:  {len(low)}")

    # Get all numeric columns, excluding metadata/label columns
    exclude = {'arousal', 'valence', 'arousal_binary', 'window_start', 'window_end',
               'label_binary', 'label_source', 'scale', 'condition', 'session'}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    results = []
    for col in numeric_cols:
        h = high[col].dropna()
        l = low[col].dropna()
        if len(h) < 5 or len(l) < 5:
            continue
        t_stat, p_val = stats.ttest_ind(h, l, equal_var=False)
        d = compute_cohens_d(h, l)
        results.append({
            'feature': col,
            'high_mean': round(h.mean(), 4),
            'high_sd': round(h.std(), 4),
            'low_mean': round(l.mean(), 4),
            'low_sd': round(l.std(), 4),
            'diff_means': round(h.mean() - l.mean(), 4),
            't_stat': round(t_stat, 3),
            'p_value': round(p_val, 6),
            'cohens_d': round(d, 4) if not np.isnan(d) else np.nan,
            'significant_p05': 'YES' if p_val < 0.05 else 'no',
            'significant_bonferroni': 'YES' if p_val < (0.05 / len(numeric_cols)) else 'no',
            'is_real_sensor': 'YES' if col in ALL_REAL_FEATURES else 'synthetic/face',
            'sensor_type': ('GSR' if col in REAL_SENSOR_FEATURES['gsr']
                           else 'HRV' if col in REAL_SENSOR_FEATURES['hrv']
                           else 'face/meta'),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('cohens_d', key=abs, ascending=False)

    # Save full table
    csv_path = os.path.join(output_dir, 'feature_significance_all.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nSaved full results: {csv_path}")

    # Print summary
    n_sig = (results_df['significant_p05'] == 'YES').sum()
    n_bonf = (results_df['significant_bonferroni'] == 'YES').sum()
    n_real_sig = ((results_df['significant_p05'] == 'YES') &
                  (results_df['is_real_sensor'] == 'YES')).sum()

    print(f"\n{'='*70}")
    print(f"SIGNIFICANCE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total features tested:            {len(results_df)}")
    print(f"Significant at p<0.05:            {n_sig}")
    print(f"Significant after Bonferroni:     {n_bonf}  (corrected threshold p < {0.05/len(results_df):.6f})")
    print(f"Real sensor features significant: {n_real_sig} of {len(ALL_REAL_FEATURES)}")
    print(f"\nTop 20 features by effect size (Cohen's d):")
    print(f"{'Feature':<45} {'d':>7} {'p':>10} {'sig?':>6} {'sensor':>12}")
    print('-'*82)
    for _, row in results_df.head(20).iterrows():
        sig = row['significant_p05']
        bonf = '(bonf)' if row['significant_bonferroni'] == 'YES' else ''
        print(f"{row['feature']:<45} {row['cohens_d']:>7.3f} {row['p_value']:>10.6f} "
              f"{sig:>6} {row['sensor_type']:>12} {bonf}")

    print(f"\nReal sensor features only:")
    real_df = results_df[results_df['is_real_sensor'] == 'YES'].copy()
    print(f"{'Feature':<25} {'d':>7} {'p':>10} {'sig?':>6} {'high mean':>10} {'low mean':>10}")
    print('-'*72)
    for _, row in real_df.iterrows():
        print(f"{row['feature']:<25} {row['cohens_d']:>7.3f} {row['p_value']:>10.6f} "
              f"{row['significant_p05']:>6} {row['high_mean']:>10.4f} {row['low_mean']:>10.4f}")

    return results_df


def plot_significance(results_df, output_dir):
    """Plot Cohen's d effect sizes for all features, coloured by sensor type."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: All features ranked by |Cohen's d| ---
    top_n = min(40, len(results_df))
    plot_df = results_df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # flip so largest is at top

    color_map = {'GSR': '#E8795A', 'HRV': '#4C72B0', 'face/meta': '#AAAAAA'}
    colors = [color_map.get(s, '#AAAAAA') for s in plot_df['sensor_type']]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    bars = ax.barh(range(len(plot_df)), plot_df['cohens_d'].abs(), color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'], fontsize=8)
    ax.axvline(0.2, color='gray', linestyle=':', alpha=0.5, label='Small effect (d=0.2)')
    ax.axvline(0.5, color='orange', linestyle=':', alpha=0.7, label='Medium effect (d=0.5)')
    ax.axvline(0.8, color='red', linestyle=':', alpha=0.7, label='Large effect (d=0.8)')
    ax.set_xlabel("|Cohen's d| (effect size)")
    ax.set_title(f"Feature Effect Sizes: High vs Low Arousal\n"
                 f"Top {top_n} features by |Cohen's d|, coloured by sensor type")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E8795A', label='GSR features'),
                        Patch(facecolor='#4C72B0', label='HRV/HR features'),
                        Patch(facecolor='#AAAAAA', label='Synthetic face features')]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][2:],
               loc='lower right', fontsize=8)

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'feature_significance_ranked.png')
    plt.savefig(path1, dpi=150)
    print(f"Saved: {path1}")
    plt.close()

    # --- Chart 2: Real sensor features only, one small subplot per feature ---
    # (previous version put all 15 features on one shared y-axis, which
    # visually flattened microsiemens-scale GSR features next to
    # millisecond-scale rr_mean -- this version gives each feature its
    # own axis so the actual high-vs-low difference is visible)
    real_df = results_df[results_df['is_real_sensor'] == 'YES'].copy()
    real_df = real_df.sort_values('cohens_d', key=abs, ascending=False)

    feature_units = {
        'gsr_min': 'uS', 'gsr_max': 'uS', 'gsr_mean': 'uS', 'gsr_std': 'uS',
        'gsr_range': 'uS', 'gsr_slope': 'uS/s', 'gsr_scl_mean': 'uS',
        'gsr_scr_amplitude': 'uS', 'gsr_peak_count': 'count',
        'hr_mean': 'BPM', 'hr_max': 'BPM', 'hr_min': 'BPM', 'hr_std': 'BPM',
        'rr_mean': 'ms', 'hrv_pnn50': '%',
    }

    n = len(real_df)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(real_df.iterrows()):
        ax = axes[i]
        means = [row['low_mean'], row['high_mean']]
        sds = [row['low_sd'], row['high_sd']]
        colors = ['#4C72B0', '#C44E52']
        bars = ax.bar(['Low', 'High'], means, yerr=sds, capsize=4,
                       color=colors, alpha=0.85)
        unit = feature_units.get(row['feature'], '')
        ax.set_title(f"{row['feature']}\nd={row['cohens_d']:.2f}"
                     f"{'  *' if row['significant_p05']=='YES' else ''}",
                     fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Real Sensor Features: High vs Low Arousal\n'
                 '(each panel on its own scale; * = significant at p<0.05; '
                 'd = Cohen\'s effect size)', fontsize=12)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'feature_significance_real_sensors.png')
    plt.savefig(path2, dpi=150)
    print(f"Saved: {path2}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="analysis_results")
    parser.add_argument("--scale", default="episodic",
                         choices=["micro", "momentary", "episodic"],
                         help="Which time scale's features to test")
    args = parser.parse_args()

    print(f"Loading {args.scale}-scale features from {args.data_dir}...")
    df = load_all_features(args.data_dir, scale=args.scale)
    if df is not None:
        results_df = run_significance_tests(df, args.output_dir)
        if results_df is not None:
            plot_significance(results_df, args.output_dir)