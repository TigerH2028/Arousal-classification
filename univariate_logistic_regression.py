"""
FILE: univariate_logistic_regression.py
PURPOSE: Perform univariate analysis on each physiological feature individually:
         (1) significance testing via Welch's t-test and Cohen's d effect size,
             using the exact same formulas AND sample as Table 4 (see
             stats_common.py) -- these are the numbers that appear as "p" and
             "Cohen's d" in Table 5;
         (2) isolated predictive capability via a standardized univariate
             logistic classifier evaluated with 5-fold stratified CV AUC/
             accuracy -- a separate quantity from (1), reported alongside it.

         CHANGED: previously used a logistic-regression Wald test (statsmodels
         Logit p-value) for significance, on a differently-filtered sample
         than Table 4. That produced numbers that looked like they should
         match Table 4 but didn't. Both scripts now share one sample
         (complete-case, listwise deletion across all 15 features) and one
         significance test (Welch's t-test), via stats_common.py.

USAGE:
    python univariate_logistic_regression.py --data_dir data/processed --output_dir analysis_results
"""

import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from stats_common import (
    cohens_d_pooled, welch_ttest, effect_size_zone, sig_stars,
    complete_case_sample,
)

# -----------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------
AROUSAL_THRESHOLD = 5.0
RANDOM_STATE = 42

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]

FEATURE_FULLNAMES = {
    'gsr_min': 'GSR Min', 'gsr_max': 'GSR Max',
    'gsr_mean': 'GSR Mean', 'gsr_std': 'GSR SD',
    'gsr_range': 'GSR Range', 'gsr_slope': 'GSR Slope',
    'gsr_scl_mean': 'SCL Mean', 'gsr_scr_amplitude': 'SCR Ampl.',
    'gsr_peak_count': 'Peak Count', 'hr_mean': 'HR Mean',
    'hr_max': 'HR Max', 'hr_min': 'HR Min', 'hr_std': 'HR SD',
    'rr_mean': 'RR Mean', 'hrv_pnn50': 'pNN50',
}

# -----------------------------------------------------------------------
# DATA LOADING -- unchanged
# -----------------------------------------------------------------------
def load_features(data_dir, scale='episodic', emotion_filter=None, max_per_session=100):
    pattern = os.path.join(data_dir, f'*/features_{scale}.csv')
    files = glob(pattern)
    if not files:
        files = glob(os.path.join(data_dir, f'**/features_{scale}.csv'), recursive=True)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if df.empty or 'arousal' not in df.columns:
                continue
            folder = os.path.basename(os.path.dirname(f))
            emotion = 'unknown'
            for key in ['anger','fear','sadness','disgust','amusement','gratitude','neutral','tenderness']:
                if key in folder.lower():
                    emotion = key
                    break
            df['emotion'] = emotion
            if emotion_filter and emotion != emotion_filter:
                continue
            if len(df) > max_per_session:
                df = df.sample(n=max_per_session, random_state=RANDOM_STATE)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['arousal_binary'] = (combined['arousal'] >= AROUSAL_THRESHOLD).astype(int)
    return combined

# -----------------------------------------------------------------------
# UNIVARIATE ANALYSIS ENGINE
# -----------------------------------------------------------------------
def run_univariate_analysis(df, output_dir, label='all_emotions'):
    clean, avail = complete_case_sample(df, REAL_SENSOR_FEATURES)
    n_dropped = len(df) - len(clean)
    y = clean['arousal_binary'].values

    print(f"\n{'='*85}")
    print(f"UNIVARIATE ANALYSIS (TABLE 5) — {label.replace('_',' ').upper()}")
    print(f"{'='*85}")
    print(f"Complete-case sample (listwise deletion across all {len(avail)} features): "
          f"N={len(clean)}  (dropped {n_dropped} of {len(df)} windows -- "
          f"same sample as Table 4 and Table 6/7)")
    print(f"High Arousal: {(y==1).sum()} | Low Arousal: {(y==0).sum()}")
    print(f"{'-'*85}")

    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for feat in avail:
        x_raw = clean[feat].values

        g1 = x_raw[y == 1]  # High arousal group
        g0 = x_raw[y == 0]  # Low arousal group

        # --- 1. Significance test: Welch's t-test + Cohen's d, matching Table 4
        #        exactly (same formulas via stats_common.py, same sample). ---
        t_stat, df_w, p_val = welch_ttest(g1, g0)
        cohens_d = cohens_d_pooled(g1, g0)
        sig_str = sig_stars(p_val)

        n1, n0 = len(g1), len(g0)
        se_d = np.sqrt((n1 + n0) / (n1 * n0) + (cohens_d ** 2) / (2 * (n1 + n0)))
        d_ci_lower = cohens_d - 1.96 * se_d
        d_ci_upper = cohens_d + 1.96 * se_d

        # --- 2. Predictive capability via 5-Fold Stratified CV (unchanged). ---
        #        This is a SEPARATE quantity from the significance test above:
        #        "how well can a logistic model rank windows using this feature
        #        alone?", not "is the group difference significant?"
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_raw.reshape(-1, 1))

        fold_aucs, fold_accs = [], []
        lr = LogisticRegression(C=1.0, class_weight='balanced', random_state=RANDOM_STATE)

        for tr_idx, te_idx in skf.split(x_scaled, y):
            X_tr, X_te = x_scaled[tr_idx], x_scaled[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            lr.fit(X_tr, y_tr)
            probs = lr.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fold_aucs.append(roc_auc_score(y_te, probs))
            fold_accs.append(accuracy_score(y_te, preds))

        cv_auc_mean = np.mean(fold_aucs)
        cv_auc_sd = np.std(fold_aucs)
        cv_acc_mean = np.mean(fold_accs)

        results.append({
            'feature': feat,
            'full_name': FEATURE_FULLNAMES.get(feat, feat),
            't_stat': t_stat,
            'df': df_w,
            'p_value': p_val,
            'sig': sig_str,
            'cohens_d': cohens_d,
            'd_zone': effect_size_zone(cohens_d, 'd'),
            'd_ci_lower': d_ci_lower,
            'd_ci_upper': d_ci_upper,
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_sd': cv_auc_sd,
            'cv_acc_mean': cv_acc_mean,
        })

    res_df = pd.DataFrame(results).sort_values(by='cv_auc_mean', ascending=False)

    # Print Formatted Results Table
    d_header = "Cohen's d (95% CI)"
    print(f"\n{'Feature':<18} {'t':>8} {'df':>7} {'p-val':>10} {'Sig':>4} {d_header:>24} {'5-Fold AUC':>14}")
    print("="*105)
    for _, r in res_df.iterrows():
        d_str = f"{r['cohens_d']:+.2f} [{r['d_ci_lower']:+.2f}, {r['d_ci_upper']:+.2f}]"
        auc_str = f"{r['cv_auc_mean']:.3f} ±{r['cv_auc_sd']:.2f}"
        print(f"{r['full_name']:<18} {r['t_stat']:>8.3f} {r['df']:>7.1f} {r['p_value']:>10.2e} {r['sig']:>4} {d_str:>24} {auc_str:>14}")
    print("="*105)
    print("Significance Legend: *** p < 0.001, ** p < 0.01, * p < 0.05, NS = Not Significant (p >= 0.05)")
    print("t, df, p, and Cohen's d use Welch's t-test on the shared complete-case sample -- identical")
    print("methodology and sample to Table 4. 5-Fold AUC/Accuracy are a separate predictive-performance")
    print("check from a standardized univariate logistic classifier, not part of the significance test.")

    # Save Results to CSV
    csv_path = os.path.join(output_dir, f'univariate_results_{label}.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")

    # Plot Visualizations
    plot_univariate_summary(res_df, output_dir, label)
    return res_df

# -----------------------------------------------------------------------
# PLOTTING FUNCTION -- unchanged aside from reading the new column names
# -----------------------------------------------------------------------
def plot_univariate_summary(res_df, output_dir, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    df_auc = res_df.sort_values(by='cv_auc_mean', ascending=True)
    colors = ['#C44E52' if p < 0.05 else '#7F7F7F' for p in df_auc['p_value']]

    bars = ax1.barh(range(len(df_auc)), df_auc['cv_auc_mean'], xerr=df_auc['cv_auc_sd'],
                    color=colors, alpha=0.85, capsize=3)
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Chance (0.5)')
    ax1.set_yticks(range(len(df_auc)))
    ax1.set_yticklabels(df_auc['full_name'], fontsize=9)
    ax1.set_xlabel('Univariate 5-Fold CV ROC AUC')
    ax1.set_title(f'Single Feature Predictive Capabilities\n({label.replace("_"," ").title()})')
    ax1.set_xlim(0.4, 0.85)
    ax1.legend(loc='lower right')

    for bar, auc, sig in zip(bars, df_auc['cv_auc_mean'], df_auc['sig']):
        if sig != 'NS':
            ax1.text(auc + 0.02, bar.get_y() + bar.get_height()/2, sig,
                     va='center', fontsize=10, fontweight='bold', color='#C44E52')

    df_d = res_df.copy()
    df_d['abs_d'] = df_d['cohens_d'].abs()
    df_d = df_d.sort_values(by='abs_d', ascending=True)
    y_pos = np.arange(len(df_d))

    ax2.barh(y_pos, df_d['abs_d'], color='#4C72B0', alpha=0.85, height=0.6)
    ax2.axvline(0.2, color='gray', linestyle=':', linewidth=1.2, label='Small (|d| = 0.2)')
    ax2.axvline(0.5, color='orange', linestyle=':', linewidth=1.2, label='Medium (|d| = 0.5)')
    ax2.axvline(0.8, color='red', linestyle=':', linewidth=1.2, label='Large (|d| = 0.8)')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_d['full_name'], fontsize=9)
    ax2.set_xlabel("|Cohen's d| (Effect Size Magnitude)")
    ax2.set_title(f'Statistical Significance & Effect Size\n({label.replace("_"," ").title()})')

    max_limit = max(0.9, df_d['abs_d'].max() + 0.1)
    ax2.set_xlim(0, max_limit)
    ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'univariate_analysis_{label}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {plot_path}")
    plt.close()

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--output_dir', default='analysis_results')
    parser.add_argument('--scale', default='episodic', choices=['micro', 'momentary', 'episodic'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset for All Emotions...")
    df_all = load_features(args.data_dir, scale=args.scale)
    if df_all is not None:
        run_univariate_analysis(df_all, args.output_dir, label='all_emotions')

    print("\nLoading dataset for Fear Only...")
    df_fear = load_features(args.data_dir, scale=args.scale, emotion_filter='fear')
    if df_fear is not None and len(df_fear) >= 20:
        run_univariate_analysis(df_fear, args.output_dir, label='fear_only')
