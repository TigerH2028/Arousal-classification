"""
FILE: visualize_feature_space.py
PURPOSE: Visualize the 15-dimensional real sensor feature space in 2D
         using two complementary approaches:

         (1) CORRELATION HEATMAP — shows relationships BETWEEN features
             (are GSR features redundant with each other? does heart rate
             cluster separately from EDA?)

         (2) PCA BIPLOT — projects all windows into 2D using Principal
             Component Analysis, colours points by arousal class (high/low),
             and overlays feature loading arrows showing which features
             drive separation along each axis

         Both plots are made for ALL EMOTIONS combined and FEAR ONLY.

WHAT IS PCA:
  Principal Component Analysis finds the directions of maximum variance
  in a high-dimensional dataset. PC1 is the single direction that
  captures the most spread in the data; PC2 captures the second most,
  perpendicular to PC1; and so on. By projecting all windows onto just
  PC1 and PC2, you can see the main structure of the data in 2D, even
  though the original data has 15 dimensions.

  The loading arrows show how much each original feature contributes to
  each principal component direction. A long arrow pointing right means
  that feature strongly drives PC1. If high-arousal (red) and low-arousal
  (blue) points separate along the direction of an arrow, that feature
  is contributing to class separation.

USAGE:
    python visualize_feature_space.py --data_dir data/processed
                                       --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

AROUSAL_THRESHOLD = 5.0

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]

# Short labels for PCA arrows (avoid overlap)
SHORT_NAMES = {
    'gsr_min':           'g_min',
    'gsr_max':           'g_max',
    'gsr_mean':          'g_mean',
    'gsr_std':           'g_std',
    'gsr_range':         'g_range',
    'gsr_slope':         'g_slope',
    'gsr_scl_mean':      'g_scl',
    'gsr_scr_amplitude': 'g_scr',
    'gsr_peak_count':    'g_peaks',
    'hr_mean':           'hr_mean',
    'hr_max':            'hr_max',
    'hr_min':            'hr_min',
    'hr_std':            'hr_std',
    'rr_mean':           'rr_mean',
    'hrv_pnn50':         'pnn50',
}

SENSOR_COLORS = {
    'gsr_min': '#E8795A', 'gsr_max': '#E8795A', 'gsr_mean': '#E8795A',
    'gsr_std': '#E8795A', 'gsr_range': '#E8795A', 'gsr_slope': '#E8795A',
    'gsr_scl_mean': '#E8795A', 'gsr_scr_amplitude': '#E8795A',
    'gsr_peak_count': '#E8795A',
    'hr_mean': '#4C72B0', 'hr_max': '#4C72B0', 'hr_min': '#4C72B0',
    'hr_std': '#4C72B0', 'rr_mean': '#4C72B0', 'hrv_pnn50': '#4C72B0',
}


# -----------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------

def load_features(data_dir, scale='episodic', emotion_filter=None,
                  max_per_session=100):
    pattern = os.path.join(data_dir, f'*/features_{scale}.csv')
    files   = glob(pattern)
    if not files:
        files = glob(os.path.join(data_dir, f'**/features_{scale}.csv'),
                     recursive=True)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if df.empty or 'arousal' not in df.columns:
                continue
            folder  = os.path.basename(os.path.dirname(f))
            emotion = 'unknown'
            for key in ['anger','fear','sadness','disgust','amusement',
                        'gratitude','neutral','tenderness']:
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
    combined['arousal_binary'] = (
        combined['arousal'] >= AROUSAL_THRESHOLD).astype(int)
    combined['arousal_label'] = combined['arousal_binary'].map(
        {1: 'High arousal', 0: 'Low arousal'})
    return combined


# -----------------------------------------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------------------------------------

def plot_correlation_heatmap(df, output_dir, label='all_emotions'):
    avail = [f for f in REAL_SENSOR_FEATURES if f in df.columns]
    corr  = df[avail].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10)

    ax.set_xticks(range(len(avail)))
    ax.set_yticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(avail, fontsize=9)

    for i in range(len(avail)):
        for j in range(len(avail)):
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6,
                    color='white' if abs(val) > 0.65 else 'black')

    # Draw sensor group boxes
    n_gsr = sum(1 for f in avail if f.startswith('gsr'))
    n_hr  = len(avail) - n_gsr
    for start, count, color, name in [
        (0, n_gsr, '#E8795A', 'GSR features'),
        (n_gsr, n_hr, '#4C72B0', 'HRV/HR features')
    ]:
        rect = plt.Rectangle((start - 0.5, start - 0.5),
                               count, count,
                               linewidth=2, edgecolor=color,
                               facecolor='none')
        ax.add_patch(rect)
        ax.text(start + count/2 - 0.5, start - 0.8, name,
                ha='center', color=color, fontsize=8, fontweight='bold')

    ax.set_title(
        f'Feature Correlation Heatmap\n({label.replace("_"," ")})\n'
        f'Red = positive correlation, Blue = negative correlation\n'
        f'Boxes outline GSR (orange) and HRV/HR (blue) sensor groups',
        fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# PCA BIPLOT
# -----------------------------------------------------------------------

def plot_pca_biplot(df, output_dir, label='all_emotions'):
    avail  = [f for f in REAL_SENSOR_FEATURES if f in df.columns]
    clean  = df[avail + ['arousal_binary', 'arousal_label']].dropna()
    X      = clean[avail].values
    y      = clean['arousal_binary'].values
    labels = clean['arousal_label'].values

    # Remove outliers beyond 3 SD per feature before PCA
    from scipy import stats as scipy_stats
    z_scores = np.abs(scipy_stats.zscore(X, axis=0))
    mask     = (z_scores < 3).all(axis=1)
    X, y, labels = X[mask], y[mask], labels[mask]

    # Scale (required: PCA is sensitive to feature scale)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    var1   = pca.explained_variance_ratio_[0] * 100
    var2   = pca.explained_variance_ratio_[1] * 100
    total  = var1 + var2

    fig, ax = plt.subplots(figsize=(11, 8))

    # Scatter points coloured by arousal class
    colors_map = {'High arousal': '#C44E52', 'Low arousal': '#4C72B0'}
    for grp_label, color in colors_map.items():
        m = labels == grp_label
        ax.scatter(X_pca[m, 0], X_pca[m, 1], alpha=0.35, s=20,
                   color=color, label=f'{grp_label} (n={m.sum()})',
                   rasterized=True)

    # Feature loading arrows
    # Scale arrows so they span ~30% of the plot range
    loadings  = pca.components_.T
    arrow_scale = 0.3 * max(
        np.abs(X_pca[:, 0]).max(),
        np.abs(X_pca[:, 1]).max()
    ) / max(np.abs(loadings).max(), 1e-6)

    for i, feat in enumerate(avail):
        lx = loadings[i, 0] * arrow_scale
        ly = loadings[i, 1] * arrow_scale
        color = SENSOR_COLORS.get(feat, 'gray')
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                   mutation_scale=12))
        offset_x = lx * 1.18
        offset_y = ly * 1.18
        ax.text(offset_x, offset_y,
                SHORT_NAMES.get(feat, feat),
                fontsize=7.5, color=color, ha='center', va='center',
                fontweight='bold')

    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

    ax.set_xlabel(f'PC1 ({var1:.1f}% of variance explained)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2:.1f}% of variance explained)', fontsize=11)
    ax.set_title(
        f'PCA Biplot: Feature Space Coloured by Arousal Class\n'
        f'({label.replace("_"," ")}, {total:.1f}% variance explained by 2 PCs)\n'
        f'Arrows = feature loading directions  '
        f'| Orange = GSR features  | Blue = HRV/HR features',
        fontsize=10)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#C44E52', label='High arousal'),
        mpatches.Patch(color='#4C72B0', label='Low arousal'),
        mpatches.Patch(color='#E8795A', label='GSR feature arrow'),
        mpatches.Patch(color='#4C72B0', alpha=0.5, label='HRV/HR feature arrow'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='best')

    # Print loadings summary
    print(f"\nPCA Variance Explained ({label}):")
    print(f"  PC1: {var1:.1f}%    PC2: {var2:.1f}%    Total: {total:.1f}%")
    print(f"\nFeature loadings on PC1 and PC2 (ranked by |PC1|):")
    load_df = pd.DataFrame(
        {'feature': avail,
         'PC1': pca.components_[0],
         'PC2': pca.components_[1]})
    load_df = load_df.reindex(
        load_df['PC1'].abs().sort_values(ascending=False).index)
    print(f"  {'Feature':<22} {'PC1':>8}  {'PC2':>8}")
    print('  ' + '-'*40)
    for _, row in load_df.iterrows():
        print(f"  {row['feature']:<22} {row['PC1']:>+8.3f}  "
              f"{row['PC2']:>+8.3f}")

    plt.tight_layout()
    path = os.path.join(output_dir, f'pca_biplot_{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
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

    # Load all emotions
    print("Loading all emotion data...")
    df_all = load_features(args.data_dir, scale=args.scale)
    if df_all is None:
        print("ERROR: No data found. Run [3] PROCESS first.")
        raise SystemExit(1)
    print(f"  {len(df_all)} windows  |  "
          f"High={df_all['arousal_binary'].sum()}  "
          f"Low={(df_all['arousal_binary']==0).sum()}")

    # Load fear only
    print("\nLoading fear-only data...")
    df_fear = load_features(args.data_dir, scale=args.scale,
                             emotion_filter='fear')
    if df_fear is not None:
        print(f"  {len(df_fear)} windows  |  "
              f"High={df_fear['arousal_binary'].sum()}  "
              f"Low={(df_fear['arousal_binary']==0).sum()}")

    # Correlation heatmaps
    print("\nGenerating correlation heatmaps...")
    plot_correlation_heatmap(df_all, args.output_dir, 'all_emotions')
    if df_fear is not None:
        plot_correlation_heatmap(df_fear, args.output_dir, 'fear_only')

    # PCA biplots
    print("\nGenerating PCA biplots...")
    plot_pca_biplot(df_all, args.output_dir, 'all_emotions')
    if df_fear is not None:
        plot_pca_biplot(df_fear, args.output_dir, 'fear_only')

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Files generated:")
    print("  heatmap_all_emotions.png")
    print("  heatmap_fear_only.png")
    print("  pca_biplot_all_emotions.png")
    print("  pca_biplot_fear_only.png")
