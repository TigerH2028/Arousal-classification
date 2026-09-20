"""
FILE: eda_ratio_threshold_analysis.py
PURPOSE: After rebuilding data with true resting baselines (from
         Baselines folder), recompute the distribution of EDA ratios,
         test whether they are normally distributed, compute two
         alternative thresholds (median vs 0-95th percentile mean),
         visualize everything, and compare how many windows fall into
         high/low arousal under each threshold.

USAGE:
    python eda_ratio_threshold_analysis.py --data_dir data/raw
                                            --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------
# 1. LOAD ALL EDA RATIOS FROM SELF_REPORT FILES
# -----------------------------------------------------------------------

def load_eda_ratios(data_dir):
    """
    Load eda_ratio values from every self_report.csv in data/raw.
    Also loads the baseline_source column so we can check how many
    sessions used the true baseline vs the fallback.
    """
    files = glob(os.path.join(data_dir, '*/self_report.csv'))
    if not files:
        files = glob(os.path.join(data_dir, '**/self_report.csv'),
                     recursive=True)

    print(f"Found {len(files)} self_report.csv files")

    all_ratios, baseline_sources, emotions = [], [], []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if 'eda_ratio' not in df.columns:
                continue
            ratios = df['eda_ratio'].dropna().tolist()
            all_ratios.extend(ratios)

            if 'baseline_source' in df.columns:
                baseline_sources.extend(df['baseline_source'].tolist())
            if 'emotion_label' in df.columns:
                emotions.extend(df['emotion_label'].tolist())
        except Exception:
            continue

    arr = np.array(all_ratios, dtype=float)
    print(f"Total EDA ratio values loaded: {len(arr)}")

    if baseline_sources:
        from collections import Counter
        src_counts = Counter(baseline_sources)
        print("\nBaseline sources used:")
        for src, count in src_counts.most_common():
            pct = count / len(baseline_sources) * 100
            print(f"  {src}: {count} ({pct:.1f}%)")

    return arr, emotions


# -----------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS + NORMALITY TESTS
# -----------------------------------------------------------------------

def describe_distribution(ratios):
    """
    Compute full descriptive statistics and run normality tests.
    """
    print("\n" + "="*65)
    print("EDA RATIO DISTRIBUTION — DESCRIPTIVE STATISTICS")
    print("="*65)
    print(f"  N total:         {len(ratios)}")
    print(f"  Min:             {ratios.min():.4f}")
    print(f"  Max:             {ratios.max():.4f}")
    print(f"  Mean (all):      {ratios.mean():.4f}")
    print(f"  Median:          {np.median(ratios):.4f}")
    print(f"  SD:              {ratios.std():.4f}")

    skew = stats.skew(ratios)
    kurt = stats.kurtosis(ratios)
    print(f"  Skewness:        {skew:.4f}  "
          f"({'right-skewed (long high tail)' if skew > 0.5 else 'left-skewed' if skew < -0.5 else 'approx. symmetric'})")
    print(f"  Kurtosis:        {kurt:.4f}  "
          f"({'heavy-tailed vs normal' if kurt > 1 else 'lighter tails than normal'})")

    print(f"\n  Percentiles:")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"    {p:3d}th: {np.percentile(ratios, p):.4f}")

    # Normality tests
    print(f"\n  NORMALITY TESTS:")

    # Shapiro-Wilk (best for n < 5000)
    sample = ratios if len(ratios) <= 5000 else np.random.choice(
        ratios, 5000, replace=False)
    sw_stat, sw_p = stats.shapiro(sample)
    print(f"  Shapiro-Wilk:    W={sw_stat:.4f}, p={sw_p:.6f}  "
          f"-> {'NOT normal (p<0.05)' if sw_p < 0.05 else 'Consistent with normal'}")

    # D'Agostino-Pearson (works well for larger samples)
    k2, dp_p = stats.normaltest(ratios)
    print(f"  D'Agostino K²:   stat={k2:.4f}, p={dp_p:.6f}  "
          f"-> {'NOT normal (p<0.05)' if dp_p < 0.05 else 'Consistent with normal'}")

    # Test if log-transformed ratios are more normal
    pos_ratios = ratios[ratios > 0]
    log_ratios = np.log(pos_ratios)
    _, log_sw_p = stats.shapiro(
        log_ratios if len(log_ratios) <= 5000
        else np.random.choice(log_ratios, 5000, replace=False))
    print(f"  Shapiro-Wilk on log(ratio): p={log_sw_p:.6f}  "
          f"-> {'Log-normal distribution likely' if log_sw_p > 0.05 else 'Also not log-normal'}")

    return skew, kurt


# -----------------------------------------------------------------------
# 3. COMPUTE BOTH THRESHOLDS
# -----------------------------------------------------------------------

def compute_thresholds(ratios):
    """
    Compute two alternative thresholds:
    (A) Median of all ratios
    (B) Mean of ratios in the 0-95th percentile range (excludes extreme outliers)
    Both are then converted to an arousal value on the 1-9 scale using
    the percentile-calibrated rescaling formula.
    """
    # Percentile anchors (will be recomputed from this data)
    p10  = np.percentile(ratios, 10)
    p50  = np.percentile(ratios, 50)   # median
    p90  = np.percentile(ratios, 90)
    p95  = np.percentile(ratios, 95)

    # Threshold A: overall median
    thresh_a_ratio = p50

    # Threshold B: mean of 0-95th percentile (clips extreme high outliers)
    ratios_0_95    = ratios[ratios <= p95]
    thresh_b_ratio = ratios_0_95.mean()

    print("\n" + "="*65)
    print("THRESHOLD COMPUTATION")
    print("="*65)
    print(f"\n  Percentile anchors for rescaling (from this data):")
    print(f"    10th percentile:  {p10:.4f}  -> arousal = 1")
    print(f"    50th percentile:  {p50:.4f}  -> arousal = 5")
    print(f"    90th percentile:  {p90:.4f}  -> arousal = 9")
    print(f"    95th percentile:  {p95:.4f}  (used to clip outliers for Threshold B)")
    print(f"\n  Threshold A — Median (all ratios):")
    print(f"    ratio threshold = {thresh_a_ratio:.4f}")
    print(f"    -> arousal = 5.0 exactly (median always maps to midpoint)")
    print(f"\n  Threshold B — Mean of 0-95th percentile range:")
    print(f"    N values in 0-95th range: {len(ratios_0_95)} "
          f"of {len(ratios)} ({len(ratios_0_95)/len(ratios)*100:.1f}%)")
    print(f"    ratio threshold = {thresh_b_ratio:.4f}")

    # Convert Threshold B ratio to arousal value
    def ratio_to_arousal(r, low=p10, mid=p50, high=p90):
        if r <= mid:
            span = mid - low
            a = 1.0 + (r - low) / span * 4.0 if span > 0 else 5.0
        else:
            span = high - mid
            a = 5.0 + (r - mid) / span * 4.0 if span > 0 else 5.0
        return float(np.clip(a, 1, 9))

    thresh_b_arousal = ratio_to_arousal(thresh_b_ratio)
    print(f"    -> arousal cutoff = {thresh_b_arousal:.3f}  "
          f"({'above 5.0 = fewer high-arousal labels' if thresh_b_arousal > 5.0 else 'below 5.0 = more high-arousal labels'})")

    return {
        'p10': p10, 'p50': p50, 'p90': p90, 'p95': p95,
        'thresh_a_ratio': thresh_a_ratio,
        'thresh_a_arousal': 5.0,
        'thresh_b_ratio': thresh_b_ratio,
        'thresh_b_arousal': thresh_b_arousal,
        'n_0_95': len(ratios_0_95),
    }, ratio_to_arousal


# -----------------------------------------------------------------------
# 4. COMPARE LABEL DISTRIBUTIONS UNDER EACH THRESHOLD
# -----------------------------------------------------------------------

def compare_label_distributions(data_dir, thresholds, ratio_to_arousal):
    """
    Load all processed arousal values and recount high/low under each
    threshold, also broken down by emotion category.
    """
    files = glob(os.path.join(
        data_dir.replace('raw', 'processed'), '*/features_episodic.csv'))
    if not files:
        files = glob('data/processed/*/features_episodic.csv')

    if not files:
        print("\nNote: No processed feature files found.")
        print("Run [3] PROCESS first, then re-run this script.")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'arousal' in df.columns:
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
                dfs.append(df[['arousal', 'session', 'emotion']].dropna())
        except Exception:
            continue

    if not dfs:
        return None

    all_df = pd.concat(dfs, ignore_index=True)
    arousal = all_df['arousal'].values

    print("\n" + "="*65)
    print("LABEL DISTRIBUTION COMPARISON")
    print("="*65)
    print(f"Total windows analyzed: {len(arousal)}")

    results = {}
    for name, cutoff in [
        ('A — Median (ratio=%.4f, arousal=5.000)' % thresholds['thresh_a_ratio'],
         thresholds['thresh_a_arousal']),
        ('B — 0-95th pct mean (ratio=%.4f, arousal=%.3f)' % (
            thresholds['thresh_b_ratio'], thresholds['thresh_b_arousal']),
         thresholds['thresh_b_arousal']),
    ]:
        n_high = (arousal >= cutoff).sum()
        n_low  = len(arousal) - n_high
        pct_high = n_high / len(arousal) * 100
        print(f"\n  Threshold {name}")
        print(f"    Arousal cutoff:  {cutoff:.3f}")
        print(f"    High arousal:    {n_high} ({pct_high:.1f}%)")
        print(f"    Low arousal:     {n_low} ({100-pct_high:.1f}%)")
        results[name] = {
            'cutoff': cutoff, 'n_high': int(n_high),
            'n_low': int(n_low), 'pct_high': round(pct_high, 1)
        }

    # Per-emotion breakdown
    print(f"\n  Per-emotion breakdown (Threshold A vs B):")
    print(f"  {'Emotion':<14} {'N':>5}  "
          f"{'%High (A)':>10}  {'%High (B)':>10}  {'Change':>8}")
    print("  " + "-"*55)
    for emotion in sorted(all_df['emotion'].unique()):
        sub = all_df[all_df['emotion'] == emotion]['arousal']
        pct_a = (sub >= thresholds['thresh_a_arousal']).mean() * 100
        pct_b = (sub >= thresholds['thresh_b_arousal']).mean() * 100
        change = pct_b - pct_a
        print(f"  {emotion:<14} {len(sub):>5}  "
              f"{pct_a:>9.1f}%  {pct_b:>9.1f}%  {change:>+7.1f}%")

    return all_df, results


# -----------------------------------------------------------------------
# 5. VISUALIZATION
# -----------------------------------------------------------------------

def plot_all(ratios, thresholds, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ta = thresholds['thresh_a_ratio']
    tb = thresholds['thresh_b_ratio']
    p95 = thresholds['p95']

    # ---- Panel 1: Histogram (clipped at 98th pct for readability) ----
    ax1 = fig.add_subplot(gs[0, 0])
    clip = np.percentile(ratios, 98)
    vis  = ratios[ratios <= clip]
    ax1.hist(vis, bins=60, color='#4C72B0', alpha=0.75, edgecolor='white')
    ax1.axvline(ta, color='orange', lw=2, ls='--',
                label=f'A: median={ta:.3f}')
    ax1.axvline(tb, color='red',    lw=2, ls='-',
                label=f'B: 0-95th mean={tb:.3f}')
    ax1.axvline(p95, color='gray',  lw=1, ls=':',
                label=f'95th pct={p95:.3f}')
    ax1.set_xlabel('EDA ratio (current / true resting baseline)')
    ax1.set_ylabel('Count')
    ax1.set_title('EDA Ratio Distribution\n(clipped at 98th pct for visibility)')
    ax1.legend(fontsize=7)

    # ---- Panel 2: Log-scale histogram ----
    ax2 = fig.add_subplot(gs[0, 1])
    pos = ratios[ratios > 0]
    ax2.hist(np.log(pos), bins=60, color='#55A868', alpha=0.75,
             edgecolor='white')
    if ta > 0:
        ax2.axvline(np.log(ta), color='orange', lw=2, ls='--',
                    label=f'A: log(median)={np.log(ta):.3f}')
    if tb > 0:
        ax2.axvline(np.log(tb), color='red', lw=2, ls='-',
                    label=f'B: log(0-95th mean)={np.log(tb):.3f}')
    ax2.set_xlabel('log(EDA ratio)')
    ax2.set_ylabel('Count')
    ax2.set_title('Log-Transformed EDA Ratio\n(bell shape = log-normal distribution)')
    ax2.legend(fontsize=7)

    # ---- Panel 3: Q-Q plot (full) ----
    ax3 = fig.add_subplot(gs[0, 2])
    trimmed = ratios[ratios <= p95]
    (osm, osr), (slope, intercept, r) = stats.probplot(
        trimmed, dist='norm', fit=True)
    ax3.scatter(osm, osr, s=3, alpha=0.3, color='#4C72B0')
    lx = np.array([osm.min(), osm.max()])
    ax3.plot(lx, slope*lx + intercept, 'r-', lw=2,
             label=f'Normal ref (r={r:.3f})')
    ax3.set_xlabel('Theoretical normal quantiles')
    ax3.set_ylabel('Sample quantiles')
    ax3.set_title('Q-Q Plot: EDA Ratio vs Normal\n'
                  '(0-95th pct values, points on line = normal)')
    ax3.legend(fontsize=8)

    # ---- Panel 4: Q-Q plot on log-transformed ratios ----
    ax4 = fig.add_subplot(gs[1, 0])
    log_trimmed = np.log(pos[pos <= p95])
    (osm2, osr2), (slope2, intercept2, r2) = stats.probplot(
        log_trimmed, dist='norm', fit=True)
    ax4.scatter(osm2, osr2, s=3, alpha=0.3, color='#55A868')
    lx2 = np.array([osm2.min(), osm2.max()])
    ax4.plot(lx2, slope2*lx2 + intercept2, 'r-', lw=2,
             label=f'Normal ref (r={r2:.3f})')
    ax4.set_xlabel('Theoretical normal quantiles')
    ax4.set_ylabel('Sample quantiles (log scale)')
    ax4.set_title('Q-Q Plot: log(EDA Ratio) vs Normal\n'
                  'Better fit here confirms log-normality')
    ax4.legend(fontsize=8)

    # ---- Panel 5: Boxplot by threshold showing split ----
    ax5 = fig.add_subplot(gs[1, 1])
    labels_a = ['High' if r >= ta else 'Low' for r in ratios]
    labels_b = ['High' if r >= tb else 'Low' for r in ratios]
    pct_high_a = labels_a.count('High') / len(labels_a) * 100
    pct_high_b = labels_b.count('High') / len(labels_b) * 100

    x    = [0, 1]
    highs = [pct_high_a, pct_high_b]
    lows  = [100-pct_high_a, 100-pct_high_b]
    ax5.bar(x, lows,  color='#4C72B0', alpha=0.85, label='Low arousal')
    ax5.bar(x, highs, bottom=lows, color='#C44E52', alpha=0.85,
            label='High arousal')
    ax5.axhline(50, color='black', ls=':', lw=1)
    for i, (h, l) in enumerate(zip(highs, lows)):
        ax5.text(i, 50+h/2, f'{h:.1f}%', ha='center', va='center',
                 color='white', fontsize=10, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([
        f'Threshold A\n(median={ta:.3f})',
        f'Threshold B\n(0-95th mean={tb:.3f})'
    ], fontsize=8)
    ax5.set_ylabel('% of windows')
    ax5.set_ylim(0, 110)
    ax5.set_title('% High vs Low Arousal\nUnder Each Threshold')
    ax5.legend(fontsize=8)

    # ---- Panel 6: Summary text ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    sw_stat, sw_p = stats.shapiro(
        ratios if len(ratios) <= 5000
        else np.random.choice(ratios, 5000, replace=False))
    _, log_sw_p = stats.shapiro(
        np.log(pos) if len(pos) <= 5000
        else np.random.choice(np.log(pos), 5000, replace=False))
    summary = (
        f"DISTRIBUTION SUMMARY\n"
        f"{'─'*30}\n"
        f"N total:          {len(ratios)}\n"
        f"Min / Max:        {ratios.min():.3f} / {ratios.max():.3f}\n"
        f"Mean (all):       {ratios.mean():.3f}\n"
        f"Median:           {np.median(ratios):.3f}\n"
        f"SD:               {ratios.std():.3f}\n"
        f"Skewness:         {stats.skew(ratios):.3f}\n"
        f"95th pct:         {p95:.3f}\n"
        f"Mean (0-95th):    {thresholds['thresh_b_ratio']:.3f}\n"
        f"\nNORMALITY\n"
        f"{'─'*30}\n"
        f"Shapiro-Wilk p:   {sw_p:.6f}\n"
        f"  {'NOT normal' if sw_p < 0.05 else 'Approx normal'}\n"
        f"log(ratio) SW p:  {log_sw_p:.6f}\n"
        f"  {'Log-normal likely' if log_sw_p > 0.05 else 'Not log-normal either'}\n"
        f"\nTHRESHOLD COMPARISON\n"
        f"{'─'*30}\n"
        f"A (median={ta:.3f}):\n"
        f"  {pct_high_a:.1f}% high / {100-pct_high_a:.1f}% low\n"
        f"B (0-95th mean={tb:.3f}):\n"
        f"  {pct_high_b:.1f}% high / {100-pct_high_b:.1f}% low\n"
        f"Change:  {pct_high_b-pct_high_a:+.1f}% high arousal"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=8.5, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle('EDA Ratio Distribution Analysis\n'
                 '(True resting baselines from POPANE Baselines folder)',
                 fontsize=13, fontweight='bold')

    path = os.path.join(output_dir, 'eda_ratio_threshold_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw')
    parser.add_argument('--output_dir', default='analysis_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ratios, emotions = load_eda_ratios(args.data_dir)
    if len(ratios) == 0:
        print("ERROR: No eda_ratio values found.")
        print("Make sure you have rebuilt data/raw with the new popane_loader.py")
        print("and that self_report.csv files contain the eda_ratio column.")
        raise SystemExit(1)

    skew, kurt = describe_distribution(ratios)
    thresholds, ratio_to_arousal = compute_thresholds(ratios)
    compare_label_distributions(args.data_dir, thresholds, ratio_to_arousal)
    plot_all(ratios, thresholds, args.output_dir)

    # Save thresholds to JSON for reference by other scripts
    out = os.path.join(args.output_dir, 'eda_ratio_thresholds.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in thresholds.items()}, f, indent=2)
    print(f"Saved thresholds: {out}")
