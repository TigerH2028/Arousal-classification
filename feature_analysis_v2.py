"""
FILE: feature_analysis_v2.py
PURPOSE: Three analyses, all using the corrected median threshold (1.359)
         from true resting baselines:

         1. FEATURE SIGNIFICANCE — recompute Welch's t-test and Cohen's d
            for all 15 real sensor features with the new binary split
            (high: arousal >= 5.0 at median ratio=1.359)

         2. FEAR-ONLY ANALYSIS — repeat all statistical tests restricted
            to windows from the Fear1 emotion category only, to see whether
            the pattern holds within a single, theoretically well-defined
            high-arousal emotion

         3. REGRESSION PLOTS — 2D scatter of each real sensor feature
            against the continuous EDA ratio (what the model ultimately
            learns from), with a regression line showing the linear
            relationship

         4. CLASSIFICATION PLOTS — same 2D scatter but with points
            coloured by binary label (high/low arousal), showing how
            cleanly each feature separates the two classes

USAGE:
    python feature_analysis_v2.py --data_dir data/processed
                                   --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# CONSTANTS — updated from true baseline analysis
# -----------------------------------------------------------------------
NEW_MEDIAN_RATIO   = 1.3585   # from eda_ratio_thresholds.json
AROUSAL_THRESHOLD  = 5.0      # arousal scale midpoint (always 5.0 for median split)
BONFERRONI_N       = 15       # number of real sensor features tested
BONFERRONI_THRESH  = 0.05 / BONFERRONI_N

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]

FEATURE_UNITS = {
    'gsr_min': 'uS',  'gsr_max': 'uS',  'gsr_mean': 'uS',
    'gsr_std': 'uS',  'gsr_range': 'uS', 'gsr_slope': 'uS/s',
    'gsr_scl_mean': 'uS', 'gsr_scr_amplitude': 'uS',
    'gsr_peak_count': 'count',
    'hr_mean': 'BPM', 'hr_max': 'BPM', 'hr_min': 'BPM',
    'hr_std': 'BPM',  'rr_mean': 'ms',  'hrv_pnn50': '%',
}

FEATURE_FULLNAMES = {
    'gsr_min':           'GSR Minimum (uS)',
    'gsr_max':           'GSR Maximum (uS)',
    'gsr_mean':          'GSR Mean (uS)',
    'gsr_std':           'GSR Std Dev (uS)',
    'gsr_range':         'GSR Range (uS)',
    'gsr_slope':         'GSR Slope (uS/s)',
    'gsr_scl_mean':      'Skin Conductance Level Mean (uS)',
    'gsr_scr_amplitude': 'SCR Amplitude (uS)',
    'gsr_peak_count':    'SCR Peak Count',
    'hr_mean':           'Heart Rate Mean (BPM)',
    'hr_max':            'Heart Rate Maximum (BPM)',
    'hr_min':            'Heart Rate Minimum (BPM)',
    'hr_std':            'Heart Rate Std Dev (BPM)',
    'rr_mean':           'RR Interval Mean (ms)',
    'hrv_pnn50':         'pNN50 (%)',
}

GSR_FEATURES = [f for f in REAL_SENSOR_FEATURES if f.startswith('gsr')]
HRV_FEATURES = [f for f in REAL_SENSOR_FEATURES if not f.startswith('gsr')]


# -----------------------------------------------------------------------
# DATA LOADING
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

            # Infer emotion from folder name
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
    combined['arousal_label'] = combined['arousal_binary'].map(
        {1: 'High arousal', 0: 'Low arousal'})

    print(f"  Total windows: {len(combined)}")
    if emotion_filter:
        print(f"  Emotion filter: {emotion_filter}")
    high = combined['arousal_binary'].sum()
    print(f"  High arousal: {high} ({high/len(combined)*100:.1f}%)")
    print(f"  Low arousal:  {len(combined)-high} "
          f"({(len(combined)-high)/len(combined)*100:.1f}%)")
    return combined


# -----------------------------------------------------------------------
# STATISTICAL SIGNIFICANCE TESTS
# -----------------------------------------------------------------------

def compute_cohens_d(h, l):
    n1, n2 = len(h), len(l)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1-1)*h.std()**2 + (n2-1)*l.std()**2) / (n1+n2-2))
    return (h.mean() - l.mean()) / pooled if pooled > 1e-10 else np.nan


def run_significance(df, label='ALL EMOTIONS'):
    high = df[df['arousal_binary'] == 1]
    low  = df[df['arousal_binary'] == 0]

    print(f"\n{'='*70}")
    print(f"FEATURE SIGNIFICANCE — {label}")
    print(f"Threshold: arousal >= {AROUSAL_THRESHOLD} "
          f"(median EDA ratio = {NEW_MEDIAN_RATIO})")
    print(f"  High: n={len(high)}  Low: n={len(low)}")
    print(f"  Bonferroni threshold: p < {BONFERRONI_THRESH:.5f} "
          f"(0.05 / {BONFERRONI_N} features)")
    print(f"{'='*70}")
    print(f"{'Feature':<22} {'d':>7} {'p':>12} {'p<.05':>6} "
          f"{'Bonf':>6} {'High mean':>10} {'Low mean':>10}")
    print('-'*70)

    results = []
    for feat in REAL_SENSOR_FEATURES:
        if feat not in df.columns:
            continue
        h_vals = high[feat].dropna()
        l_vals = low[feat].dropna()
        if len(h_vals) < 3 or len(l_vals) < 3:
            continue

        t, p = stats.ttest_ind(h_vals, l_vals, equal_var=False)
        d = compute_cohens_d(h_vals, l_vals)

        sig05  = 'YES' if p < 0.05 else 'no'
        sig_bf = 'YES' if p < BONFERRONI_THRESH else 'no'

        print(f"{feat:<22} {d:>7.3f} {p:>12.6f} {sig05:>6} "
              f"{sig_bf:>6} {h_vals.mean():>10.4f} {l_vals.mean():>10.4f}")

        results.append({
            'feature': feat, 'cohens_d': d, 'p_value': p,
            'sig_p05': sig05, 'sig_bonferroni': sig_bf,
            'high_mean': h_vals.mean(), 'high_sd': h_vals.std(),
            'low_mean': l_vals.mean(), 'low_sd': l_vals.std(),
            'n_high': len(h_vals), 'n_low': len(l_vals),
        })

    print('-'*70)
    results_df = pd.DataFrame(results).sort_values('cohens_d', key=abs,
                                                     ascending=False)
    n_sig   = (results_df['sig_p05'] == 'YES').sum()
    n_bonf  = (results_df['sig_bonferroni'] == 'YES').sum()
    print(f"Significant at p<0.05:        {n_sig}/{len(results_df)} "
          f"({n_sig/len(results_df)*100:.0f}%)")
    print(f"Significant after Bonferroni: {n_bonf}/{len(results_df)} "
          f"({n_bonf/len(results_df)*100:.0f}%)")
    return results_df


# -----------------------------------------------------------------------
# REGRESSION PLOTS — feature vs EDA ratio (continuous)
# -----------------------------------------------------------------------

def plot_regression(df, output_dir, label='all'):
    """
    For each real sensor feature: scatter plot of feature value (y-axis)
    against the continuous arousal score (x-axis, which is the rescaled
    EDA ratio). Fit a linear regression line. This shows whether each
    feature has a linear relationship with the underlying physiology measure.
    """
    avail = [f for f in REAL_SENSOR_FEATURES if f in df.columns
             and 'arousal' in df.columns]
    if not avail:
        print("No features available for regression plots")
        return

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)

    arousal_vals = df['arousal'].values

    for i, feat in enumerate(avail):
        ax = axes[i]
        feat_vals = df[feat].dropna()
        common_idx = df[feat].dropna().index.intersection(
            df['arousal'].dropna().index)
        x = df.loc[common_idx, 'arousal'].values
        y = df.loc[common_idx, feat].values

        # Remove extreme outliers (beyond 3 SD) for cleaner plot
        x_z = np.abs(stats.zscore(x))
        y_z = np.abs(stats.zscore(y))
        mask = (x_z < 3) & (y_z < 3)
        x_c, y_c = x[mask], y[mask]

        # Scatter
        ax.scatter(x_c, y_c, alpha=0.2, s=8, color='#4C72B0')

        # Regression line
        if len(x_c) > 5:
            slope, intercept, r, p, se = stats.linregress(x_c, y_c)
            xline = np.linspace(x_c.min(), x_c.max(), 100)
            ax.plot(xline, slope*xline + intercept, 'r-', lw=2,
                    label=f'r={r:.3f}, p={p:.4f}')
            sig = '*' if p < 0.05 else ''
            ax.set_title(f"{feat}\nr={r:.3f}{sig}  slope={slope:.4f}",
                         fontsize=9)
        else:
            ax.set_title(feat, fontsize=9)

        ax.axvline(AROUSAL_THRESHOLD, color='green', lw=1, ls='--',
                   alpha=0.5, label='threshold')
        ax.set_xlabel('Arousal (rescaled EDA ratio, 1-9)', fontsize=8)
        ax.set_ylabel(FEATURE_FULLNAMES.get(feat, feat), fontsize=7)
        ax.legend(fontsize=6)

    for j in range(len(avail), len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'Regression: Sensor Features vs Continuous Arousal Score\n'
        f'({label}, threshold={AROUSAL_THRESHOLD}, '
        f'median EDA ratio={NEW_MEDIAN_RATIO})\n'
        f'r = Pearson correlation, * = p<0.05, outliers >3SD removed',
        fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'regression_features_vs_arousal_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# CLASSIFICATION PLOTS — feature vs arousal, coloured by binary label
# -----------------------------------------------------------------------

def plot_classification(df, output_dir, label='all'):
    """
    For each real sensor feature: scatter plot of feature value (y-axis)
    against the continuous arousal score (x-axis), with points coloured
    by binary label (high=red, low=blue). A vertical threshold line shows
    where the split occurs. This shows how cleanly each feature separates
    the two arousal classes.
    """
    avail = [f for f in REAL_SENSOR_FEATURES if f in df.columns]
    if not avail:
        return

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).reshape(-1)

    colors = {1: '#C44E52', 0: '#4C72B0'}
    group_labels = {1: 'High arousal', 0: 'Low arousal'}

    for i, feat in enumerate(avail):
        ax = axes[i]
        common_idx = df[feat].dropna().index.intersection(
            df['arousal'].dropna().index)
        x = df.loc[common_idx, 'arousal'].values
        y = df.loc[common_idx, feat].values
        b = df.loc[common_idx, 'arousal_binary'].values

        # Remove outliers for cleaner plot
        x_z = np.abs(stats.zscore(x))
        y_z = np.abs(stats.zscore(y))
        mask = (x_z < 3) & (y_z < 3)
        x_c, y_c, b_c = x[mask], y[mask], b[mask]

        for grp in [0, 1]:
            m = b_c == grp
            ax.scatter(x_c[m], y_c[m], alpha=0.3, s=8,
                       color=colors[grp], label=group_labels[grp])

        # Threshold line
        ax.axvline(AROUSAL_THRESHOLD, color='black', lw=1.5, ls='--',
                   label=f'threshold={AROUSAL_THRESHOLD}')

        # Compute mean separation between groups
        h_mean = y_c[b_c == 1].mean() if (b_c == 1).sum() > 0 else np.nan
        l_mean = y_c[b_c == 0].mean() if (b_c == 0).sum() > 0 else np.nan

        if not np.isnan(h_mean) and not np.isnan(l_mean):
            ax.axhline(h_mean, color='#C44E52', lw=1, ls=':',
                       alpha=0.7)
            ax.axhline(l_mean, color='#4C72B0', lw=1, ls=':',
                       alpha=0.7)
            diff = h_mean - l_mean
            ax.set_title(
                f"{feat}\nHigh mean={h_mean:.3f}  Low mean={l_mean:.3f}"
                f"  diff={diff:+.3f}", fontsize=8)
        else:
            ax.set_title(feat, fontsize=8)

        ax.set_xlabel('Arousal score (1-9)', fontsize=8)
        ax.set_ylabel(FEATURE_FULLNAMES.get(feat, feat), fontsize=7)
        ax.legend(fontsize=6, markerscale=2)

    for j in range(len(avail), len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'Classification: Sensor Features vs Binary Arousal Label\n'
        f'({label}, vertical line = threshold at arousal={AROUSAL_THRESHOLD})\n'
        f'Dotted horizontal lines = group means',
        fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir,
                        f'classification_features_vs_label_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# EFFECT SIZE SUMMARY CHART
# -----------------------------------------------------------------------

def plot_effect_sizes(results_all, results_fear, output_dir):
    """
    Side-by-side Cohen's d bars: all emotions vs fear only.
    Makes it easy to see which features have stronger effects within fear
    compared to the overall dataset.
    """
    features = results_all['feature'].tolist()
    d_all  = results_all.set_index('feature')['cohens_d']
    d_fear = results_fear.set_index('feature')['cohens_d'] if results_fear is not None else None

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    bars1 = ax.bar(x - width/2, [abs(d_all.get(f, 0)) for f in features],
                    width, label='All emotions', color='#4C72B0', alpha=0.85)

    if d_fear is not None:
        bars2 = ax.bar(x + width/2,
                        [abs(d_fear.get(f, 0)) for f in features],
                        width, label='Fear only', color='#C44E52', alpha=0.85)

    ax.axhline(0.2, color='gray', ls=':', lw=1, label='Small (d=0.2)')
    ax.axhline(0.5, color='orange', ls=':', lw=1, label='Medium (d=0.5)')
    ax.axhline(0.8, color='red', ls=':', lw=1, label='Large (d=0.8)')

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("|Cohen's d| (effect size)")
    ax.set_title(
        f"Effect Sizes: All Emotions vs Fear Only\n"
        f"(New median threshold, arousal >= {AROUSAL_THRESHOLD}, "
        f"EDA ratio median = {NEW_MEDIAN_RATIO})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, 'effect_sizes_all_vs_fear.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--output_dir', default='analysis_results')
    parser.add_argument('--scale', default='episodic',
                        choices=['micro', 'momentary', 'episodic'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. All emotions ----
    print("\nLoading ALL emotion data...")
    df_all = load_features(args.data_dir, scale=args.scale)
    if df_all is None:
        print("ERROR: No processed data found. Run [3] PROCESS first.")
        raise SystemExit(1)

    results_all = run_significance(df_all, label='ALL EMOTIONS')
    results_all.to_csv(
        os.path.join(args.output_dir, 'feature_significance_new_threshold.csv'),
        index=False)

    # ---- 2. Fear only ----
    print("\n\nLoading FEAR-ONLY data...")
    df_fear = load_features(args.data_dir, scale=args.scale,
                             emotion_filter='fear')
    results_fear = None
    if df_fear is not None and len(df_fear) > 10:
        results_fear = run_significance(df_fear, label='FEAR ONLY')
        results_fear.to_csv(
            os.path.join(args.output_dir,
                         'feature_significance_fear_only.csv'),
            index=False)
    else:
        print("Not enough Fear data for separate analysis.")

    # ---- 3. Regression plots ----
    print("\nGenerating regression plots...")
    plot_regression(df_all, args.output_dir, label='all_emotions')
    if df_fear is not None and len(df_fear) > 10:
        plot_regression(df_fear, args.output_dir, label='fear_only')

    # ---- 4. Classification plots ----
    print("Generating classification plots...")
    plot_classification(df_all, args.output_dir, label='all_emotions')
    if df_fear is not None and len(df_fear) > 10:
        plot_classification(df_fear, args.output_dir, label='fear_only')

    # ---- 5. Effect size comparison chart ----
    print("Generating effect size comparison...")
    plot_effect_sizes(results_all, results_fear, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Files generated:")
    print("  feature_significance_new_threshold.csv  (all emotions)")
    print("  feature_significance_fear_only.csv      (fear only)")
    print("  regression_features_vs_arousal_all_emotions.png")
    print("  regression_features_vs_arousal_fear_only.png")
    print("  classification_features_vs_label_all_emotions.png")
    print("  classification_features_vs_label_fear_only.png")
    print("  effect_sizes_all_vs_fear.png")
