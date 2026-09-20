"""
FILE: tuned_classification_models.py

PURPOSE:
    Build and compare four classifiers with proper hyperparameter tuning:
    (1) Logistic Regression with L1 penalty
    (2) Logistic Regression with L2 penalty
    (3) Support Vector Machine (SVM / SVC)
    (4) Gradient Boosting (scikit-learn GradientBoostingClassifier)

    All models use:
    - Consistent z-score normalization (fit on train only, never on test)
    - Nested cross-validation: outer 5-fold for evaluation,
      inner 3-fold for hyperparameter search
    - Stratified folds to preserve class balance
    - class_weight='balanced' for LR and SVM

DATA PIPELINE:
    Raw POPANE → artifact removal → person-relative EDA normalization
    → arousal labeling (median split at 5.0) → signal processing
    → 15 real sensor features (9 GSR, 6 HRV) → NaN removal
    → z-score normalization → stratified nested CV

USAGE:
    python tuned_classification_models.py
           --data_dir data/processed
           --output_dir analysis_results
           --scale episodic
"""

import os
import json
import argparse
import textwrap
import warnings
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------

AROUSAL_THRESHOLD = 5.0
RANDOM_STATE = 42
N_OUTER_FOLDS = 5     # outer CV: for evaluation
N_INNER_FOLDS = 3     # inner CV: for hyperparameter tuning

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]

FEATURE_FULLNAMES = {
    'gsr_min': 'GSR Min',     'gsr_max': 'GSR Max',
    'gsr_mean': 'GSR Mean',   'gsr_std': 'GSR SD',
    'gsr_range': 'GSR Range', 'gsr_slope': 'GSR Slope',
    'gsr_scl_mean': 'SCL Mean',
    'gsr_scr_amplitude': 'SCR Ampl.',
    'gsr_peak_count': 'Peak Count',
    'hr_mean': 'HR Mean',     'hr_max': 'HR Max',
    'hr_min': 'HR Min',       'hr_std': 'HR SD',
    'rr_mean': 'RR Mean',     'hrv_pnn50': 'pNN50',
}

# -----------------------------------------------------------------------
# HYPERPARAMETER GRIDS
# -----------------------------------------------------------------------

LR_L2_GRID = {
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs'],
}

LR_L1_GRID = {
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__penalty': ['l1'],
    'classifier__solver': ['liblinear'],
}

SVM_GRID = {
    'classifier__C': [0.1, 1.0, 10.0, 100.0],
    'classifier__kernel': ['rbf', 'linear'],
    'classifier__gamma': ['scale', 'auto'],
}

GB_GRID = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [2, 3, 5],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__min_samples_leaf': [5, 10],
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
                df = df.sample(n=max_per_session, random_state=RANDOM_STATE)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    combined['arousal_binary'] = (
        combined['arousal'] >= AROUSAL_THRESHOLD).astype(int)
    return combined


# -----------------------------------------------------------------------
# NESTED CV WITH HYPERPARAMETER TUNING
# -----------------------------------------------------------------------

def run_nested_cv(df, model_name, param_grid, base_estimator,
                  output_dir, label='all_emotions'):
    avail = [f for f in REAL_SENSOR_FEATURES if f in df.columns]
    clean = df[avail + ['arousal_binary']].dropna()
    X     = clean[avail].values
    y     = clean['arousal_binary'].values
    n_dropped = len(df) - len(clean)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}  |  DATASET: {label.replace('_',' ')}")
    print(f"{'='*70}")
    print(f"Windows loaded:   {len(df)}")
    print(f"Windows dropped:  {n_dropped}  (NaN in >=1 feature)")
    print(f"Windows used:     {len(y)}  "
          f"(High={(y==1).sum()}, Low={(y==0).sum()})")
    print(f"Outer CV:         {N_OUTER_FOLDS}-fold stratified (evaluation)")
    print(f"Inner CV:         {N_INNER_FOLDS}-fold (hyperparameter search)")
    print(f"Z-score:          fit on TRAIN fold only, applied to TEST")
    
    n_combinations = 1
    for v in param_grid.values():
        n_combinations *= len(v)
    print(f"Param grid:       {n_combinations} combinations to search "
          f"({N_INNER_FOLDS}-fold inner CV = "
          f"{n_combinations * N_INNER_FOLDS} fits per outer fold)")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', base_estimator),
    ])

    outer_skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)

    fold_aucs, fold_accs, fold_f1p, fold_f1n = [], [], [], []
    fold_cms, fold_fprs, fold_tprs = [], [], []
    fold_best_params = []
    all_probs, all_true = [], []

    print(f"\nPER-FOLD RESULTS (with best hyperparameters found per fold):")
    print(f"  {'Fold':>5}  {'AUC':>7}  {'Acc':>7}  {'F1+':>7}  "
          f"{'F1-':>7}  Best params (top 2)")
    print('  ' + '-'*75)

    for fold, (tr_idx, te_idx) in enumerate(outer_skf.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=inner_skf,
            scoring='roc_auc', n_jobs=-1, refit=True
        )
        grid_search.fit(X_tr, y_tr)

        probs = grid_search.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        fa   = roc_auc_score(y_te, probs)
        facc = accuracy_score(y_te, preds)
        ff1p = f1_score(y_te, preds, pos_label=1, zero_division=0)
        ff1n = f1_score(y_te, preds, pos_label=0, zero_division=0)
        fcm  = confusion_matrix(y_te, preds)
        fpr_i, tpr_i, _ = roc_curve(y_te, probs)

        fold_aucs.append(fa)
        fold_accs.append(facc)
        fold_f1p.append(ff1p)
        fold_f1n.append(ff1n)
        fold_cms.append(fcm)
        fold_fprs.append(fpr_i)
        fold_tprs.append(tpr_i)
        fold_best_params.append(grid_search.best_params_)
        all_probs.extend(probs.tolist())
        all_true.extend(y_te.tolist())

        bp = {k.replace('classifier__',''): v
              for k, v in grid_search.best_params_.items()}
        bp_str = '  '.join(f'{k}={v}' for k, v in list(bp.items())[:2])
        print(f"  {fold:>5}  {fa:>7.3f}  {facc:>7.3f}  "
              f"{ff1p:>7.3f}  {ff1n:>7.3f}  {bp_str}")

    print('  ' + '-'*75)
    print(f"  {'Mean':>5}  {np.mean(fold_aucs):>7.3f}  "
          f"{np.mean(fold_accs):>7.3f}  {np.mean(fold_f1p):>7.3f}  "
          f"{np.mean(fold_f1n):>7.3f}")
    print(f"  {'SD':>5}  {np.std(fold_aucs):>7.3f}  "
          f"{np.std(fold_accs):>7.3f}  {np.std(fold_f1p):>7.3f}  "
          f"{np.std(fold_f1n):>7.3f}")

    pooled_auc   = roc_auc_score(all_true, all_probs)
    pooled_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    pooled_acc   = accuracy_score(all_true, pooled_preds)
    pooled_f1p   = f1_score(all_true, pooled_preds, pos_label=1, zero_division=0)
    pooled_f1n   = f1_score(all_true, pooled_preds, pos_label=0, zero_division=0)
    pooled_cm    = confusion_matrix(all_true, pooled_preds)
    tn, fp, fn, tp = pooled_cm.ravel()

    print(f"\nPOOLED RESULTS ({len(all_true)} windows, each predicted once):")
    print(f"  Pooled AUC:       {pooled_auc:.3f}  "
          f"(mean fold={np.mean(fold_aucs):.3f}, SD={np.std(fold_aucs):.3f})")
    print(f"  Pooled Accuracy:  {pooled_acc:.3f}")
    print(f"  Sensitivity:      {tp/(tp+fn)*100:.1f}%")
    print(f"  Specificity:      {tn/(tn+fp)*100:.1f}%")

    print(f"\nHYPERPARAMETER SELECTION ACROSS FOLDS:")
    all_params = {}
    for bp in fold_best_params:
        for k, v in bp.items():
            k_short = k.replace('classifier__', '')
            if k_short not in all_params:
                all_params[k_short] = []
            all_params[k_short].append(v)
    for param, values in all_params.items():
        vc = Counter(values)
        print(f"  {param:<20}: {dict(vc)}  (most common: {vc.most_common(1)[0][0]})")

    results = {
        'model': model_name, 'label': label,
        'pooled_auc': pooled_auc, 'mean_fold_auc': np.mean(fold_aucs),
        'auc_sd': np.std(fold_aucs), 'pooled_acc': pooled_acc,
        'f1_pos': pooled_f1p, 'f1_neg': pooled_f1n,
        'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
        'n_total': len(y), 'n_high': int((y==1).sum()),
        'n_low': int((y==0).sum()), 'n_dropped': n_dropped,
        'per_fold_auc': fold_aucs,
        'best_params_per_fold': [
            {k.replace('classifier__',''): v for k, v in bp.items()}
            for bp in fold_best_params
        ],
    }

    _plot_results(model_name, label, fold_fprs, fold_tprs, fold_aucs,
                  pooled_cm, pooled_auc, pooled_acc, tp, tn, fp, fn,
                  all_true, all_probs, fold_best_params, output_dir)

    return results


def _plot_results(model_name, label, fold_fprs, fold_tprs, fold_aucs,
                  pooled_cm, pooled_auc, pooled_acc, tp, tn, fp, fn,
                  all_true, all_probs, fold_best_params, output_dir):

    fig = plt.figure(figsize=(21, 6))

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1.1, 0.9, 1.9],
        wspace=0.45
    )

    # Panel 1: ROC curves
    ax1 = fig.add_subplot(gs[0])
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fold_fprs)))

    for i, (fpr_i, tpr_i, auc_i) in enumerate(zip(fold_fprs, fold_tprs, fold_aucs), 1):
        ax1.plot(
            fpr_i, tpr_i,
            color=colors[i-1], lw=1.2, alpha=0.75,
            label=f'Fold {i} AUC={auc_i:.3f}'
        )

    fpr_p, tpr_p, _ = roc_curve(all_true, all_probs)
    ax1.plot(
        fpr_p, tpr_p, 'r-', lw=2.5,
        label=f'Pooled AUC={pooled_auc:.3f}\n(SD={np.std(fold_aucs):.3f})'
    )
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax1.set_xlabel('False Positive Rate (1-Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title(f'ROC Curves: Per Fold + Pooled\n{model_name} | {label.replace("_"," ")}')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)

    # Panel 2: Confusion Matrix
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(pooled_cm, cmap='Blues')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Pred Low', 'Pred High'])
    ax2.set_yticklabels(['Actual Low', 'Actual High'])

    labels = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            value = pooled_cm[r, c]
            ax2.text(
                c, r, f'{labels[r][c]}\n{value}',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if value > pooled_cm.max()*0.7 else 'black'
            )

    ax2.set_title(
        f'Pooled CM (sum of {len(fold_aucs)} folds)\n'
        f'AUC={pooled_auc:.3f}  Acc={pooled_acc:.3f}\n'
        f'Sens={tp/(tp+fn)*100:.1f}%  Spec={tn/(tn+fp)*100:.1f}%'
    )

    # Panel 3: Hyperparameter table
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')

    all_params = {}
    for bp in fold_best_params:
        for k, v in bp.items():
            k_short = k.replace('classifier__', '')
            if k_short not in all_params:
                all_params[k_short] = []
            all_params[k_short].append(v)

    table_data = []
    for param, values in all_params.items():
        vc = Counter(values)
        most_common = vc.most_common(1)[0][0]
        fold_string_raw = ", ".join([f'F{i+1}:{v}' for i, v in enumerate(values)])
        fold_string = "\n".join(textwrap.wrap(fold_string_raw, width=34))
        table_data.append([param, fold_string, str(most_common)])

    if table_data:
        tbl = ax3.table(
            cellText=table_data,
            colLabels=['Hyperparameter', 'Outer Fold Selection', 'Most Common'],
            loc='center', cellLoc='left'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for key, cell in tbl.get_celld().items():
            row, col = key
            n_lines = cell.get_text().get_text().count('\n') + 1
            cell.set_height(0.12 * n_lines)
            if col == 0:
                cell.set_width(0.30)
            elif col == 1:
                cell.set_width(0.50)
            elif col == 2:
                cell.set_width(0.20)

        ax3.set_title(
            'Hyperparameter Selection Across Folds\n(Inner 3-fold GridSearch)',
            fontsize=10, pad=15
        )

    fig.suptitle(
        f'{model_name}: Nested CV Results\n'
        f'Outer: {len(fold_aucs)}-fold evaluation | '
        f'Inner: 3-fold GridSearch | '
        f'Z-score fit on train only | '
        f'class_weight=balanced',
        fontsize=12, y=1.02
    )

    plt.tight_layout()
    slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = os.path.join(output_dir, f'tuned_{slug}_{label}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# -----------------------------------------------------------------------
# COMPARISON PLOT
# -----------------------------------------------------------------------

def plot_comparison(all_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, dataset_label in zip(axes, ['all_emotions', 'fear_only']):
        subset = [r for r in all_results if r['label'] == dataset_label]
        if not subset:
            ax.set_title(f'No data: {dataset_label}')
            continue

        names  = [r['model'] for r in subset]
        aucs   = [r['pooled_auc'] for r in subset]
        sds    = [r['auc_sd'] for r in subset]
        colors = ['#4C72B0', '#4C72B0', '#8172B0', '#E8795A', '#55A868', '#937860'][:len(names)]

        x = np.arange(len(names))
        bars = ax.bar(x, aucs, yerr=sds, capsize=5, color=colors, alpha=0.85, width=0.6)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.015,
                    f'{auc:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5, label='Chance (0.5)')
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Pooled AUC (error bars = fold SD)')
        ax.set_title(f'{dataset_label.replace("_"," ").title()}\nNested CV AUC Comparison')
        ax.legend(fontsize=8)

    fig.suptitle(
        'Model Comparison: Tuned Classifiers\n'
        'Pooled AUC from outer 5-fold CV  |  Hyperparameters tuned by '
        'inner 3-fold GridSearch  |  Z-score normalization throughout',
        fontsize=11
    )
    plt.tight_layout()
    path = os.path.join(output_dir, 'tuned_model_comparison.png')
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

    print("="*70)
    print("TUNED CLASSIFICATION PIPELINE")
    print("="*70)
    print(f"Scale:       {args.scale} (60-second windows)")
    print(f"Threshold:   arousal >= {AROUSAL_THRESHOLD} = high arousal")
    print(f"Outer folds: {N_OUTER_FOLDS} (evaluation)")
    print(f"Inner folds: {N_INNER_FOLDS} (hyperparameter tuning)")
    print(f"Scoring:     ROC AUC (inner CV selects highest AUC)")

    print("\nLoading data...")
    df_all  = load_features(args.data_dir, scale=args.scale)
    df_fear = load_features(args.data_dir, scale=args.scale, emotion_filter='fear')

    if df_all is None:
        print("ERROR: No data found. Run preprocessing pipeline first.")
        raise SystemExit(1)

    datasets = [('all_emotions', df_all)]
    if df_fear is not None and len(df_fear) >= 30:
        datasets.append(('fear_only', df_fear))

    all_results = []

    for dataset_label, df in datasets:
        # --- Model 1: LR with L2 (tuned C) ---
        lr_l2 = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        )
        res = run_nested_cv(
            df, 'LR-L2 (tuned)', LR_L2_GRID, lr_l2,
            args.output_dir, dataset_label
        )
        all_results.append(res)

        # --- Model 2: LR with L1 (tuned C) ---
        lr_l1 = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        )
        res = run_nested_cv(
            df, 'LR-L1 (tuned)', LR_L1_GRID, lr_l1,
            args.output_dir, dataset_label
        )
        all_results.append(res)

        # --- Model 3: SVM (tuned) ---
        svm = SVC(
            probability=True, random_state=RANDOM_STATE, class_weight='balanced'
        )
        res = run_nested_cv(
            df, 'SVM (tuned)', SVM_GRID, svm,
            args.output_dir, dataset_label
        )
        all_results.append(res)

        # --- Model 4: Gradient Boosting (tuned) ---
        gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        res = run_nested_cv(
            df, 'GradBoost (tuned)', GB_GRID, gb,
            args.output_dir, dataset_label
        )
        all_results.append(res)

    plot_comparison(all_results, args.output_dir)

    print(f"\n{'='*78}")
    print("FINAL SUMMARY: All Tuned Models")
    print(f"{'='*78}")
    print(f"{'Model':<22} {'Dataset':<15} {'Pooled AUC':>11} "
          f"{'AUC SD':>8} {'Acc':>7} {'Sens':>7} {'Spec':>7}")
    print('-'*78)
    for r in all_results:
        print(f"{r['model']:<22} {r['label']:<15} "
              f"{r['pooled_auc']:>11.3f} {r['auc_sd']:>8.3f} "
              f"{r['pooled_acc']:>7.3f} "
              f"{r['sensitivity']:>7.3f} {r['specificity']:>7.3f}")
    print('='*78)

    out = os.path.join(args.output_dir, 'tuned_model_results.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved: {out}")