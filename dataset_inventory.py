"""
FILE: dataset_inventory.py
PURPOSE: Scan every subfolder of your popane_raw directory -- including
         emotion categories you didn't use in training -- and report:
         (1) How many CSV files each category contains
         (2) How many have valid (non-missing) EDA and ECG columns
         (3) What the EDA range and baseline look like per category
         (4) Which studies/sensor configurations each category comes from

         This gives you a complete picture of what data is available
         and what you chose not to use, which is important for
         your methods section and for future work recommendations.

USAGE:
    python dataset_inventory.py --input_dir popane_raw
                                 --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 1000  # Hz


def check_file_validity(csv_path):
    """
    Read one POPANE CSV and return quality metrics.
    Returns a dict of results, or None if the file can't be read.
    """
    try:
        df = pd.read_csv(csv_path, comment='#', header=0)
    except Exception as e:
        return {'valid': False, 'reason': str(e), 'path': csv_path}

    result = {
        'valid': True,
        'path': csv_path,
        'filename': os.path.basename(csv_path),
        'n_rows': len(df),
        'duration_sec': round(len(df) / SAMPLE_RATE, 1),
        'columns': list(df.columns),
    }

    # Study ID from filename
    stem = Path(csv_path).stem
    parts = stem.split('_')
    result['study'] = parts[0] if parts[0].startswith('S') else 'unknown'

    # EDA check
    if 'EDA' in df.columns:
        eda = df['EDA'].dropna()
        eda_valid = eda[eda > 0]
        result['has_eda'] = True
        result['eda_n_valid'] = len(eda_valid)
        result['eda_pct_valid'] = round(len(eda_valid) / max(len(eda), 1) * 100, 1)
        result['eda_mean'] = round(eda_valid.mean(), 3) if len(eda_valid) > 0 else np.nan
        result['eda_range'] = round(eda_valid.max() - eda_valid.min(), 3) if len(eda_valid) > 0 else np.nan
        result['eda_baseline'] = round(eda_valid.iloc[:5000].median(), 3) if len(eda_valid) >= 100 else np.nan
    else:
        result['has_eda'] = False
        result['eda_n_valid'] = 0
        result['eda_pct_valid'] = 0

    # ECG check
    if 'ECG' in df.columns:
        ecg = df['ECG'].dropna()
        result['has_ecg'] = True
        result['ecg_n_valid'] = len(ecg)
        result['ecg_pct_valid'] = round(len(ecg) / max(len(df), 1) * 100, 1)
    else:
        result['has_ecg'] = False
        result['ecg_n_valid'] = 0
        result['ecg_pct_valid'] = 0

    # affect/self-report check
    if 'affect' in df.columns:
        affect = df['affect'].dropna()
        result['has_affect'] = True
        result['affect_mean'] = round(affect.mean(), 2) if len(affect) > 0 else np.nan
    else:
        result['has_affect'] = False
        result['affect_mean'] = np.nan

    # Usability: a file is "usable" if it has EDA AND duration >= 60 seconds
    result['usable_for_pipeline'] = (
        result['has_eda'] and
        result['eda_pct_valid'] > 50 and
        result['duration_sec'] >= 60
    )

    return result


def scan_all_folders(input_dir):
    """Scan all emotion subfolders and check every CSV file."""
    subfolders = sorted([f.path for f in os.scandir(input_dir) if f.is_dir()])
    print(f"Found {len(subfolders)} subfolders in {input_dir}\n")

    all_results = []
    folder_summaries = []

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        csv_files = sorted(glob(os.path.join(folder, '**', '*.csv'), recursive=True))

        if not csv_files:
            folder_summaries.append({
                'category': folder_name,
                'n_files': 0,
                'n_valid_eda': 0,
                'n_valid_ecg': 0,
                'n_usable': 0,
                'pct_usable': 0,
                'studies': '',
                'avg_duration_sec': np.nan,
                'avg_eda_mean': np.nan,
            })
            continue

        print(f"  {folder_name}: checking {len(csv_files)} files...")
        folder_results = []
        for f in csv_files:
            r = check_file_validity(f)
            r['category'] = folder_name
            all_results.append(r)
            folder_results.append(r)

        valid = [r for r in folder_results if r['valid']]
        usable = [r for r in folder_results if r.get('usable_for_pipeline', False)]
        studies = sorted(set(r.get('study', '?') for r in valid))
        avg_dur = np.mean([r['duration_sec'] for r in valid]) if valid else np.nan
        avg_eda = np.mean([r['eda_mean'] for r in valid
                            if not np.isnan(r.get('eda_mean', np.nan))]) if valid else np.nan
        n_valid_eda = sum(1 for r in folder_results if r.get('has_eda', False))
        n_valid_ecg = sum(1 for r in folder_results if r.get('has_ecg', False))

        folder_summaries.append({
            'category': folder_name,
            'n_files': len(csv_files),
            'n_valid_eda': n_valid_eda,
            'n_valid_ecg': n_valid_ecg,
            'n_usable': len(usable),
            'pct_usable': round(len(usable) / max(len(csv_files), 1) * 100, 1),
            'studies': ', '.join(studies),
            'avg_duration_sec': round(avg_dur, 1) if not np.isnan(avg_dur) else np.nan,
            'avg_eda_mean': round(avg_eda, 3) if not np.isnan(avg_eda) else np.nan,
        })

    return pd.DataFrame(all_results), pd.DataFrame(folder_summaries)


def print_and_save_summary(summary_df, output_dir, used_categories=None):
    """Print the inventory table and flag which categories were used in training."""
    if used_categories is None:
        used_categories = {'anger1', 'fear1', 'sadness1', 'disgust',
                            'amusement1', 'gratitude', 'neutral1', 'tenderness1'}

    lines = []
    lines.append('='*90)
    lines.append('POPANE DATASET INVENTORY: ALL CATEGORIES')
    lines.append('='*90)
    lines.append(f"{'Category':<30} {'Files':>6} {'EDA OK':>7} {'ECG OK':>7} "
                 f"{'Usable':>7} {'%Use':>6} {'AvgDur':>8} {'AvgEDA':>8} {'Studies':<20} {'Used?'}")
    lines.append('-'*90)

    total_files = total_usable = 0
    for _, row in summary_df.iterrows():
        used = 'YES' if row['category'].lower() in used_categories else 'no'
        dur = f"{row['avg_duration_sec']:.0f}s" if not pd.isna(row['avg_duration_sec']) else '—'
        eda = f"{row['avg_eda_mean']:.2f}" if not pd.isna(row['avg_eda_mean']) else '—'
        lines.append(
            f"{row['category']:<30} {row['n_files']:>6} {row['n_valid_eda']:>7} "
            f"{row['n_valid_ecg']:>7} {row['n_usable']:>7} {row['pct_usable']:>5.0f}% "
            f"{dur:>8} {eda:>8} {row['studies'][:20]:<20} {used}"
        )
        total_files += row['n_files']
        total_usable += row['n_usable']

    lines.append('-'*90)
    lines.append(f"{'TOTAL':<30} {total_files:>6} {'':>7} {'':>7} {total_usable:>7}")
    lines.append('='*90)
    lines.append(f"\nUsable = has EDA (>50% non-missing) AND duration >= 60 seconds")
    lines.append(f"Used? = included in model training for this project")

    text = '\n'.join(lines)
    print('\n' + text)

    path = os.path.join(output_dir, 'dataset_inventory.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSaved: {path}")

    csv_path = os.path.join(output_dir, 'dataset_inventory.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_inventory(summary_df, output_dir, used_categories=None):
    """Bar chart of file counts per category, coloured by whether it was used."""
    if used_categories is None:
        used_categories = {'anger1', 'fear1', 'sadness1', 'disgust',
                            'amusement1', 'gratitude', 'neutral1', 'tenderness1'}

    summary_df = summary_df[summary_df['n_files'] > 0].copy()
    summary_df = summary_df.sort_values('n_files', ascending=True)

    colors = ['#55A868' if c.lower() in used_categories else '#AAAAAA'
               for c in summary_df['category']]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(summary_df) * 0.35)))

    # Left: total files
    ax = axes[0]
    ax.barh(range(len(summary_df)), summary_df['n_files'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df['category'], fontsize=9)
    ax.set_xlabel('Number of CSV files')
    ax.set_title('Total Files per Category')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#55A868', label='Used in training'),
                        Patch(facecolor='#AAAAAA', label='Not used')],
               fontsize=8)

    # Right: usable files
    ax = axes[1]
    ax.barh(range(len(summary_df)), summary_df['n_usable'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df['category'], fontsize=9)
    ax.set_xlabel('Number of usable files (EDA valid, ≥60s)')
    ax.set_title('Usable Files per Category\n(has valid EDA AND ≥60s duration)')

    plt.suptitle('POPANE Dataset Inventory: All Emotion Categories',
                  fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'dataset_inventory_chart.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                         help="Top-level popane_raw folder")
    parser.add_argument("--output_dir", default="analysis_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Scanning all POPANE folders (this may take a few minutes)...")
    all_results_df, summary_df = scan_all_folders(args.input_dir)

    # Save full per-file results
    full_path = os.path.join(args.output_dir, 'dataset_inventory_perfile.csv')
    all_results_df.to_csv(full_path, index=False)
    print(f"\nPer-file results saved: {full_path}")

    print_and_save_summary(summary_df, args.output_dir)
    plot_inventory(summary_df, args.output_dir)

    # Quick quality summary
    if len(all_results_df) > 0:
        usable = all_results_df[all_results_df.get('usable_for_pipeline', False) == True]
        print(f"\nOVERALL QUALITY SUMMARY")
        print(f"  Total files scanned: {len(all_results_df)}")
        print(f"  Files with EDA:      {all_results_df['has_eda'].sum()}")
        print(f"  Files with ECG:      {all_results_df['has_ecg'].sum()}")
        print(f"  Usable files:        {all_results_df.get('usable_for_pipeline', pd.Series([False]*len(all_results_df))).sum()}")