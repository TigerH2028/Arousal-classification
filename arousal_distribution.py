"""
FILE: arousal_distribution.py
PURPOSE: Visualize the distribution of physiology-derived arousal values:
         (1) across the whole dataset as a histogram/KDE
         (2) within each emotion category as overlaid box plots
         (3) how the binary high/low split falls within each category

         This answers the question: "does our physiology-derived arousal
         label actually behave differently across emotion categories, or
         do all categories look the same?"

USAGE:
    python arousal_distribution.py --data_dir data/processed
                                    --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

AROUSAL_THRESHOLD = 5.0

EMOTION_COLORS = {
    'anger':      '#C44E52',
    'fear':       '#DD8452',
    'disgust':    '#937860',
    'sadness':    '#4C72B0',
    'amusement':  '#55A868',
    'gratitude':  '#8172B2',
    'neutral':    '#AAAAAA',
    'tenderness': '#64B5CD',
    'excitement': '#BCB800',
    'threat':     '#8C2D04',
    'baseline':   '#CCCCCC',
}


def load_arousal_and_condition(data_dir, scale='episodic', max_per_session=50):
    """Load arousal values and emotion/condition labels from processed files."""
    pattern = os.path.join(data_dir, f'*/features_{scale}.csv')
    files = glob(pattern)
    if not files:
        files = glob(os.path.join(data_dir, f'**/features_{scale}.csv'), recursive=True)

    print(f"Found {len(files)} {scale} feature files")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8')
            if df.empty or 'arousal' not in df.columns:
                continue
            if len(df) > max_per_session:
                df = df.sample(n=max_per_session, random_state=42)

            folder = os.path.basename(os.path.dirname(f))
            df['session'] = folder

            # Extract emotion from condition column if available,
            # otherwise infer from folder name
            if 'condition' in df.columns:
                df['emotion'] = df['condition'].str.lower().str.strip()
            else:
                emotion = 'unknown'
                for key in EMOTION_COLORS:
                    if key in folder.lower():
                        emotion = key
                        break
                df['emotion'] = emotion

            rows.append(df[['arousal', 'emotion', 'session']].dropna())
        except Exception:
            continue

    if not rows:
        print("ERROR: No valid data found.")
        return None

    combined = pd.concat(rows, ignore_index=True)
    combined['arousal_binary'] = (combined['arousal'] >= AROUSAL_THRESHOLD).astype(int)
    combined['arousal_label'] = combined['arousal_binary'].map({1: 'High', 0: 'Low'})
    print(f"Total windows loaded: {len(combined)}")
    print(f"Binary split: {combined['arousal_binary'].value_counts().to_dict()}")
    return combined


def plot_overall_distribution(df, output_dir):
    """Histogram + KDE of arousal across the full dataset, with threshold line."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram of all arousal values
    ax = axes[0]
    bins = np.linspace(1, 9, 33)
    ax.hist(df['arousal'], bins=bins, color='#4C72B0', alpha=0.75, edgecolor='white')
    ax.axvline(AROUSAL_THRESHOLD, color='red', linewidth=2, linestyle='--',
                label=f'Threshold (={AROUSAL_THRESHOLD})')
    low_pct = (df['arousal'] < AROUSAL_THRESHOLD).mean() * 100
    high_pct = 100 - low_pct
    ax.text(2.0, ax.get_ylim()[1]*0.9, f'Low arousal\n{low_pct:.1f}%',
             ha='center', color='#4C72B0', fontsize=10)
    ax.text(7.0, ax.get_ylim()[1]*0.9, f'High arousal\n{high_pct:.1f}%',
             ha='center', color='#C44E52', fontsize=10)
    ax.set_xlabel('Arousal value (1-9, physiology-derived)')
    ax.set_ylabel('Count (windows)')
    ax.set_title('Overall Arousal Distribution\n(physiology-derived, all windows)')
    ax.legend()

    # Right: box plot per emotion category
    ax = axes[1]
    emotions = sorted(df['emotion'].unique())
    data_per_emotion = [df.loc[df['emotion'] == e, 'arousal'].values for e in emotions]
    colors = [EMOTION_COLORS.get(e, '#888888') for e in emotions]

    bp = ax.boxplot(data_per_emotion, patch_artist=True, showmeans=True,
                     tick_labels=emotions)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(AROUSAL_THRESHOLD, color='red', linewidth=1.5, linestyle='--',
                label='Threshold')
    ax.set_xticklabels(emotions, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Arousal value (1-9)')
    ax.set_title('Arousal Distribution by Emotion Category\n'
                 '(orange line=median, green triangle=mean)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'arousal_distribution_overall.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_category_binary_split(df, output_dir):
    """Stacked bar chart showing what fraction of each category is high vs low arousal."""
    emotions = sorted(df['emotion'].unique())

    fracs_high, fracs_low, counts = [], [], []
    for e in emotions:
        sub = df[df['emotion'] == e]
        counts.append(len(sub))
        fracs_high.append((sub['arousal_binary'] == 1).mean())
        fracs_low.append((sub['arousal_binary'] == 0).mean())

    x = np.arange(len(emotions))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, fracs_low,  label='Low arousal',  color='#4C72B0', alpha=0.85)
    ax.bar(x, fracs_high, bottom=fracs_low, label='High arousal', color='#C44E52', alpha=0.85)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5,
                label='50% line')

    for i, (fh, n) in enumerate(zip(fracs_high, counts)):
        ax.text(i, 1.02, f'n={n}', ha='center', fontsize=8)
        if fh > 0.05:
            ax.text(i, 1 - fh/2, f'{fh*100:.0f}%', ha='center',
                     va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=35, ha='right')
    ax.set_ylabel('Fraction of windows')
    ax.set_ylim(0, 1.12)
    ax.set_title('Fraction of High vs Low Arousal Windows per Emotion Category\n'
                 '(n = number of windows in that category)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'arousal_binary_split_by_category.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def print_category_stats(df, output_dir):
    """Print and save per-category arousal descriptive statistics."""
    lines = []
    lines.append('='*72)
    lines.append('AROUSAL DISTRIBUTION BY EMOTION CATEGORY')
    lines.append('='*72)
    lines.append(f"{'Emotion':<14} {'N':>5} {'Mean':>7} {'SD':>7} {'Median':>8} "
                 f"{'%High':>7} {'%Low':>7}")
    lines.append('-'*72)

    for emotion in sorted(df['emotion'].unique()):
        sub = df[df['emotion'] == emotion]['arousal']
        n = len(sub)
        pct_high = (sub >= AROUSAL_THRESHOLD).mean() * 100
        pct_low = 100 - pct_high
        lines.append(f"{emotion:<14} {n:>5} {sub.mean():>7.2f} {sub.std():>7.2f} "
                     f"{sub.median():>8.2f} {pct_high:>7.1f}% {pct_low:>7.1f}%")

    lines.append('-'*72)
    sub_all = df['arousal']
    pct_high = (sub_all >= AROUSAL_THRESHOLD).mean() * 100
    lines.append(f"{'OVERALL':<14} {len(sub_all):>5} {sub_all.mean():>7.2f} "
                 f"{sub_all.std():>7.2f} {sub_all.median():>8.2f} "
                 f"{pct_high:>7.1f}% {100-pct_high:>7.1f}%")
    lines.append('='*72)

    text = '\n'.join(lines)
    print('\n' + text)
    path = os.path.join(output_dir, 'arousal_distribution_stats.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="analysis_results")
    parser.add_argument("--scale", default="episodic",
                         choices=["micro", "momentary", "episodic"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_arousal_and_condition(args.data_dir, scale=args.scale)
    if df is not None:
        print_category_stats(df, args.output_dir)
        plot_overall_distribution(df, args.output_dir)
        plot_category_binary_split(df, args.output_dir)