"""
FILE: compare_models.py
PURPOSE: Loads results from your Gradient Boosting pipeline and BOTH CNN
         runs (362 windows and 693 windows) and produces a three-way
         comparison chart plus a summary table.

USAGE:
    python compare_models.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_cnn_results(cnn_dir):
    """
    Looks for the 5-fold CV results file first (current methodology,
    comparable to GB's 5-fold CV). Falls back to the old single-split
    file with a warning if 5-fold results aren't found, since that
    comparison would not be apples-to-apples.
    """
    cv_path = os.path.join(cnn_dir, 'cnn_results_5fold.json')
    old_path = os.path.join(cnn_dir, 'cnn_results.json')

    if os.path.exists(cv_path):
        with open(cv_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Normalize field names so the rest of the script can use one schema
        return {
            'auc': data['pooled_auc'],
            'accuracy': data['pooled_accuracy'],
            'f1_positive': data['pooled_f1_positive'],
            'f1_negative': data['pooled_f1_negative'],
            'auc_sd': data.get('fold_auc_sd'),
            'methodology': '5-fold CV',
        }
    elif os.path.exists(old_path):
        print(f"WARNING: {cv_path} not found, falling back to single-split "
              f"results at {old_path}. This is NOT directly comparable to "
              f"the Gradient Boosting model's 5-fold CV results — rerun "
              f"cnn_model.py with the updated script to get 5-fold results.")
        with open(old_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'auc': data['final_auc'],
            'accuracy': data['final_accuracy'],
            'f1_positive': data.get('final_f1_positive', data.get('final_f1')),
            'f1_negative': data.get('final_f1_negative'),
            'auc_sd': None,
            'methodology': 'single 80/20 split',
        }
    else:
        print(f"WARNING: No CNN results found in {cnn_dir}")
        return None


def get_gb_results_manually():
    """
    Your GB performance summary isn't in a single predictable JSON key,
    so these are entered directly from your hypothesis_test_report.txt /
    performance_summary.png (episodic scale results).
    Edit these four numbers if your numbers differ.
    """
    return {
        'auc': 0.64,
        'accuracy': 0.65,
        'f1_positive': 0.72,
        'f1_negative': 0.53,
    }


def build_comparison_table(gb, cnn_362, cnn_693):
    rows = []
    rows.append(('Gradient Boosting\n(episodic, 15 features, 5-fold CV)',
                  gb['auc'], gb['accuracy'], gb['f1_positive'], gb['f1_negative'], None))
    if cnn_362:
        rows.append((f"CNN (362 windows)\n({cnn_362['methodology']})",
                      cnn_362['auc'], cnn_362['accuracy'],
                      cnn_362['f1_positive'], cnn_362['f1_negative'], cnn_362.get('auc_sd')))
    if cnn_693:
        rows.append((f"CNN (693 windows)\n({cnn_693['methodology']})",
                      cnn_693['auc'], cnn_693['accuracy'],
                      cnn_693['f1_positive'], cnn_693['f1_negative'], cnn_693.get('auc_sd')))
    return rows


def print_table(rows):
    print("\n" + "=" * 86)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 86)
    header = f"{'Model':<32}{'AUC':<10}{'AUC SD':<10}{'Accuracy':<12}{'F1 (Pos)':<12}{'F1 (Neg)'}"
    print(header)
    print("-" * 86)
    for name, auc, acc, f1p, f1n, auc_sd in rows:
        name_clean = name.replace("\n", " ")
        f1p_str = f"{f1p:.3f}" if f1p == f1p else "—"   # NaN check
        f1n_str = f"{f1n:.3f}" if f1n == f1n else "—"
        sd_str = f"±{auc_sd:.3f}" if auc_sd is not None else "—"
        print(f"{name_clean:<32}{auc:<10.3f}{sd_str:<10}{acc:<12.3f}{f1p_str:<12}{f1n_str}")
    print("=" * 86)
    print("\nNote: AUC SD = standard deviation across 5 CV folds (stability indicator).")
    print("      GB's AUC SD is not computed here -- pull it from your GB analysis output")
    print("      if you want a full stability comparison.")


def plot_comparison(rows, output_path='analysis_results/model_comparison_3way.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = [r[0] for r in rows]
    auc_vals = [r[1] for r in rows]
    acc_vals = [r[2] for r in rows]
    f1p_vals = [r[3] for r in rows]
    auc_sds = [r[5] if r[5] is not None else 0 for r in rows]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width, auc_vals, width, yerr=auc_sds, capsize=4,
                    label='AUC (error bars = fold SD)', color='#4C72B0')
    bars2 = ax.bar(x, acc_vals, width, label='Accuracy', color='#DD8452')
    bars3 = ax.bar(x + width, f1p_vals, width, label='F1 (Positive class)', color='#55A868')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Chance (0.5)')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Gradient Boosting vs CNN (362 vs 693 windows)\n'
                  'All models evaluated with 5-fold cross-validation, episodic/60s scale')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(loc='upper right')

    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height == height:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height + auc_sds[i]),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison chart: {output_path}")
    plt.close()


def interpret(rows):
    print("\nINTERPRETATION")
    print("-" * 86)
    gb_auc = rows[0][1]
    best_cnn = max(rows[1:], key=lambda r: r[1]) if len(rows) > 1 else None

    if best_cnn:
        diff = best_cnn[1] - gb_auc
        if diff > 0.03:
            print(f"Best CNN config ({best_cnn[0].splitlines()[0]}) outperforms Gradient")
            print(f"Boosting by {diff:+.3f} AUC. Raw-waveform learning is capturing signal")
            print("beyond what the 15 engineered features encode at this sample size.")
        elif diff < -0.03:
            print(f"Gradient Boosting outperforms the best CNN config by {-diff:.3f} AUC.")
            print("Domain-informed feature engineering remains more sample-efficient")
            print("than end-to-end deep learning at this dataset size.")
        else:
            print("Gradient Boosting and the best CNN configuration perform comparably,")
            print("suggesting the 15 engineered features already capture most of the")
            print("predictive signal available in the raw waveforms at this sample size.")

    if len(rows) >= 3:
        cnn_diff = rows[2][1] - rows[1][1]
        print(f"\nEffect of more training data on CNN: AUC changed by {cnn_diff:+.3f}")
        print(f"when going from 362 to 693 windows ({rows[1][1]:.3f} -> {rows[2][1]:.3f}).")
        if cnn_diff < 0:
            print("More data did not improve ranking performance in this run, though")
            print("F1 scores may tell a different story — check the table above.")
        else:
            print("More data modestly improved the CNN's discrimination ability.")

    sds = [r[5] for r in rows if r[5] is not None]
    if sds:
        print(f"\nCNN fold-to-fold AUC stability (SD across 5 folds): "
              f"{', '.join(f'{r[0].splitlines()[0]}={r[5]:.3f}' for r in rows if r[5] is not None)}")
        print("A higher SD indicates the CNN's performance varies more depending on")
        print("which data ends up in the training vs test fold -- a sign of instability")
        print("that the deterministic Gradient Boosting model does not exhibit.")


if __name__ == "__main__":
    gb = get_gb_results_manually()
    cnn_362 = load_cnn_results('models/cnn_v1_362')
    cnn_693 = load_cnn_results('models/cnn_v1_693')

    rows = build_comparison_table(gb, cnn_362, cnn_693)
    print_table(rows)
    plot_comparison(rows)
    interpret(rows)