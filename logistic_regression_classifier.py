"""
FILE: logistic_regression_classifier.py
PURPOSE: Build a logistic regression model to classify high vs low arousal
         from the 15 real sensor features, with:

         - Stratified 5-fold cross-validation (same methodology as the
           Gradient Boosting model, enabling direct AUC comparison)
         - Evaluation: AUC, accuracy, F1 (positive and negative class),
           confusion matrix
         - Coefficient plot showing which features drive the model
         - ROC curve comparing LR against the GB baseline

         Run separately for ALL EMOTIONS and FEAR ONLY.

WHY LOGISTIC REGRESSION:
  Logistic regression is a linear classifier — it learns a weighted sum
  of features and converts it to a probability via the sigmoid function.
  It is the standard first-step classifier for two reasons:
    (1) Interpretability: each coefficient directly tells you how much a
        one-unit increase in a feature changes the log-odds of high arousal
    (2) Baseline comparison: if LR matches GB in AUC, the relationship
        is largely linear; if GB >> LR, tree-based non-linearity adds value

WHY STRATIFICATION:
  StratifiedKFold ensures each fold preserves the same high/low arousal
  ratio as the full dataset. Without stratification, a fold could by chance
  contain mostly one class, making AUC estimates unreliable. This is
  especially important here because the class ratio (e.g. 82%/18% in fear)
  is already imbalanced.

WHY class_weight='balanced':
  With imbalanced classes (more high arousal than low), a naive model learns
  to predict "high" for everything and achieves high accuracy but zero recall
  on the minority class. Setting class_weight='balanced' tells scikit-learn
  to weight each class inversely proportional to its frequency, so the model
  is equally penalized for missing either class.

USAGE:
    python logistic_regression_classifier.py --data_dir data/processed
                                              --output_dir analysis_results
"""

import numpy as np
import pandas as pd
import os
import argparse
from glob import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

AROUSAL_THRESHOLD = 5.0

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

# Gradient Boosting baseline from earlier analysis (for comparison table)
GB_RESULTS = {
    'all_emotions': {'auc': 0.640, 'auc_sd': 0.010, 'acc': 0.650,
                     'f1_pos': 0.720, 'f1_neg': 0.530},
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
    return combined


# -----------------------------------------------------------------------
# LOGISTIC REGRESSION WITH STRATIFIED 5-FOLD CV
# -----------------------------------------------------------------------

def run_logistic_regression(df, output_dir, label='all_emotions'):
    """
    OBJECTIVE: Build a logistic regression classifier to predict binary
    arousal (high=1 / low=0) from the 15 real sensor features.

    WHAT WE ARE TRYING TO PROVE OR DISPROVE:
    (1) Whether a LINEAR model can separate high from low arousal using
        these features. If LR AUC is close to Gradient Boosting AUC,
        the relationship is largely linear and GB's non-linear complexity
        adds little value. If GB >> LR, non-linearity matters.
    (2) Which features the LINEAR model assigns the strongest weights to
        (coefficients), after controlling for all other features
        simultaneously -- unlike univariate significance tests which look
        at one feature at a time.
    (3) Whether the model generalises across different data splits
        (fold-to-fold AUC stability), or whether it overfits to
        specific windows.

    NOTE ON WINDOW COUNT vs EXPECTED:
    The script uses df.dropna() before fitting, which removes any window
    that has a missing value in ANY of the 15 features. The difference
    between the loaded window count (e.g. 600) and the fitted window
    count (e.g. 560) represents windows where at least one feature
    could not be computed -- typically because that 60-second window
    had insufficient valid signal for the signal_processor to extract
    a reliable value (e.g. too many artifact-corrupted EDA samples,
    or too few detected ECG beats to compute HRV). This is normal and
    expected with real physiological data.
    """
    avail = [f for f in REAL_SENSOR_FEATURES if f in df.columns]
    clean = df[avail + ['arousal_binary']].dropna()
    X     = clean[avail].values
    y     = clean['arousal_binary'].values

    n_dropped = len(df) - len(clean)

    print(f"\n{'='*70}")
    print(f"LOGISTIC REGRESSION — {label.replace('_',' ').upper()}")
    print(f"{'='*70}")
    print(f"Windows loaded:          {len(df)}")
    print(f"Windows dropped (NaN):   {n_dropped}  "
          f"(missing value in >=1 feature after signal processing)")
    print(f"Windows used for model:  {len(y)}")
    print(f"High arousal (label=1):  {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"Low arousal  (label=0):  {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"Features:                {len(avail)}")
    print(f"\nMODEL HYPERPARAMETERS (no tuning performed):")
    print(f"  Algorithm:       Logistic Regression (linear classifier)")
    print(f"  Regularisation:  L2 penalty (shrinks coefficients toward 0)")
    print(f"  C (strength):    1.0  (default; smaller=stronger regularisation)")
    print(f"  class_weight:    balanced  (weights classes by 1/frequency)")
    print(f"  Solver:          lbfgs (default, works well for L2 + small data)")
    print(f"  max_iter:        1000  (ensures convergence)")
    print(f"  random_state:    42   (reproducibility)")
    print(f"  Feature scaling: z-score (StandardScaler, required for LR)")
    print(f"\nVALIDATION:")
    print(f"  Method: 5-fold STRATIFIED cross-validation")
    print(f"  Each fold: train on 80% ({int(len(y)*0.8)} windows), "
          f"test on 20% (~{int(len(y)*0.2)} windows)")
    print(f"  Stratified: each fold preserves the "
          f"{(y==1).mean()*100:.0f}%/{(y==0).mean()*100:.0f}% class ratio")
    print(f"  No hyperparameter tuning: C=1.0 used in all folds without search")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                              class_weight='balanced')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_aucs, fold_accs, fold_f1p, fold_f1n = [], [], [], []
    fold_cms, fold_fprs, fold_tprs = [], [], []
    all_probs, all_true = [], []

    print(f"\nPER-FOLD RESULTS:")
    print(f"  {'Fold':>5}  {'Train N':>8}  {'Test N':>7}  "
          f"{'AUC':>7}  {'Acc':>7}  {'F1+':>7}  {'F1-':>7}")
    print('  ' + '-'*58)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_scaled, y), 1):
        X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        lr.fit(X_tr, y_tr)
        probs = lr.predict_proba(X_te)[:, 1]
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
        all_probs.extend(probs.tolist())
        all_true.extend(y_te.tolist())

        print(f"  {fold:>5}  {len(tr_idx):>8}  {len(te_idx):>7}  "
              f"{fa:>7.3f}  {facc:>7.3f}  {ff1p:>7.3f}  {ff1n:>7.3f}")

    print('  ' + '-'*58)
    print(f"  {'Mean':>5}  {'':>8}  {'':>7}  "
          f"{np.mean(fold_aucs):>7.3f}  {np.mean(fold_accs):>7.3f}  "
          f"{np.mean(fold_f1p):>7.3f}  {np.mean(fold_f1n):>7.3f}")
    print(f"  {'SD':>5}  {'':>8}  {'':>7}  "
          f"{np.std(fold_aucs):>7.3f}  {np.std(fold_accs):>7.3f}  "
          f"{np.std(fold_f1p):>7.3f}  {np.std(fold_f1n):>7.3f}")

    # Pooled metrics — each window predicted exactly once
    pooled_auc   = roc_auc_score(all_true, all_probs)
    pooled_preds = [1 if p >= 0.5 else 0 for p in all_probs]
    pooled_acc   = accuracy_score(all_true, pooled_preds)
    pooled_f1p   = f1_score(all_true, pooled_preds, pos_label=1, zero_division=0)
    pooled_f1n   = f1_score(all_true, pooled_preds, pos_label=0, zero_division=0)
    pooled_cm    = confusion_matrix(all_true, pooled_preds)
    tn, fp, fn, tp = pooled_cm.ravel()

    print(f"\nPOOLED RESULTS:")
    print(f"  Each of the {len(all_true)} windows was predicted exactly ONCE")
    print(f"  (in its held-out fold). Pooled metrics use all predictions together.")
    print(f"  Pooled AUC:       {pooled_auc:.3f}  "
          f"(mean fold AUC={np.mean(fold_aucs):.3f}, SD={np.std(fold_aucs):.3f})")
    print(f"  Pooled Accuracy:  {pooled_acc:.3f}  "
          f"= ({tn}+{tp}) / {len(all_true)}  [NOT the mean of fold accuracies]")
    print(f"  Pooled F1 (pos):  {pooled_f1p:.3f}")
    print(f"  Pooled F1 (neg):  {pooled_f1n:.3f}")
    print(f"\nPOOLED CONFUSION MATRIX (sum of all 5 folds' test-set predictions):")
    print(f"  This IS the aggregated sum — not averaged.")
    print(f"  The accuracy ({pooled_acc:.3f}) and ROC curve shown are POOLED,")
    print(f"  not the mean of per-fold values.")
    print(f"                     Pred Low    Pred High")
    print(f"  Actual Low         {tn:8d}     {fp:8d}  "
          f"(specificity={tn/(tn+fp)*100:.1f}%)")
    print(f"  Actual High        {fn:8d}     {tp:8d}  "
          f"(sensitivity={tp/(tp+fn)*100:.1f}%)")

    # Coefficients from full-data fit
    lr_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                  class_weight='balanced')
    lr_full.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        'feature': avail,
        'full_name': [FEATURE_FULLNAMES.get(f, f) for f in avail],
        'coefficient': lr_full.coef_[0]
    }).sort_values('coefficient', ascending=True)

    print(f"\nCOEFFICIENTS (fit on ALL {len(y)} windows, NOT a CV fold):")
    print(f"  Features were z-scored before fitting, so coefficients are")
    print(f"  directly comparable in magnitude across features.")
    for _, row in coef_df.sort_values('coefficient', key=abs,
                                       ascending=False).iterrows():
        bar = '█' * int(abs(row['coefficient']) * 8)
        sign = '+' if row['coefficient'] > 0 else '-'
        print(f"  {row['feature']:<22} {row['coefficient']:>+8.4f}  {bar}")

    # ---- PLOT 1: Per-fold ROC curves ----
    fig1, ax = plt.subplots(figsize=(7, 6))
    fold_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    for i, (fpr_i, tpr_i, fa_i) in enumerate(
            zip(fold_fprs, fold_tprs, fold_aucs), 1):
        ax.plot(fpr_i, tpr_i, color=fold_colors[i-1], lw=1.3, alpha=0.8,
                label=f'Fold {i}  AUC={fa_i:.3f}')
    fpr_pool, tpr_pool, _ = roc_curve(all_true, all_probs)
    ax.plot(fpr_pool, tpr_pool, 'r-', lw=2.5,
             label=f'Pooled  AUC={pooled_auc:.3f}  (SD={np.std(fold_aucs):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Chance')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title(
        f'ROC Curves: Per Fold + Pooled\n'
        f'({label.replace("_"," ")}, 5-fold stratified CV)\n'
        f'Light blue lines = individual folds  |  Red = pooled',
        fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    plt.tight_layout()
    path1 = os.path.join(output_dir, f'lr_roc_perfold_{label}.png')
    plt.savefig(path1, dpi=150)
    print(f"\nSaved: {path1}")
    plt.close()

    # ---- PLOT 2: Per-fold confusion matrices ----
    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes_flat = axes.flatten()

    max_val = max(cm.max() for cm in fold_cms)
    for i, (fcm, fa) in enumerate(zip(fold_cms, fold_aucs)):
        ax = axes_flat[i]
        ax.imshow(fcm, cmap='Blues', vmin=0, vmax=max_val)
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred\nLow','Pred\nHigh'], fontsize=8)
        ax.set_yticklabels(['Act.\nLow','Act.\nHigh'], fontsize=8)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(fcm[r,c]), ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         color='white' if fcm[r,c]>max_val*0.5 else 'black')
        n_fold = fcm.sum()
        acc_f  = (fcm[0,0]+fcm[1,1]) / n_fold
        sens_f = fcm[1,1]/(fcm[1,0]+fcm[1,1]) if (fcm[1,0]+fcm[1,1])>0 else 0
        spec_f = fcm[0,0]/(fcm[0,0]+fcm[0,1]) if (fcm[0,0]+fcm[0,1])>0 else 0
        ax.set_title(f'Fold {i+1}  (n={n_fold})\n'
                     f'AUC={fa:.3f}  Acc={acc_f:.3f}\n'
                     f'Sens={sens_f:.2f}  Spec={spec_f:.2f}',
                     fontsize=8)

    ax = axes_flat[5]
    ax.imshow(pooled_cm, cmap='Reds', vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred\nLow','Pred\nHigh'], fontsize=8)
    ax.set_yticklabels(['Act.\nLow','Act.\nHigh'], fontsize=8)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(pooled_cm[r,c]), ha='center', va='center',
                     fontsize=13, fontweight='bold',
                     color='white' if pooled_cm[r,c]>pooled_cm.max()*0.5
                     else 'black')
    ax.set_title(f'POOLED (n={len(all_true)})\n'
                 f'AUC={pooled_auc:.3f}  Acc={pooled_acc:.3f}\n'
                 f'Sens={tp/(tp+fn)*100:.1f}%  Spec={tn/(tn+fp)*100:.1f}%',
                 fontsize=8)

    fig2.suptitle(
        f'Confusion Matrices: Per Fold + Pooled\n'
        f'({label.replace("_"," ")}, 5-fold stratified CV)\n'
        f'Blue panels = individual folds  |  Red panel = aggregated sum\n'
        f'Sens = sensitivity (% high correctly identified)  '
        f'Spec = specificity (% low correctly identified)',
        fontsize=10)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f'lr_cm_perfold_{label}.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()

    # ---- PLOT 3: Summary (pooled ROC + pooled CM + coefficients) ----
    fig3 = plt.figure(figsize=(16, 5))
    gs3  = fig3.add_gridspec(1, 3, wspace=0.35)

    ax1 = fig3.add_subplot(gs3[0])
    ax1.plot(fpr_pool, tpr_pool, '#C44E52', lw=2.5,
              label=f'LR Pooled AUC={pooled_auc:.3f}\n'
                    f'Mean fold={np.mean(fold_aucs):.3f} '
                    f'(SD={np.std(fold_aucs):.3f})')
    if label in GB_RESULTS:
        ax1.axhline(GB_RESULTS[label]['auc'], color='#4C72B0',
                    lw=1.5, ls='--',
                    label=f"GB AUC={GB_RESULTS[label]['auc']:.3f} (ref)")
    ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Chance')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Pooled ROC\n({label.replace("_"," ")})')
    ax1.legend(fontsize=8); ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)

    ax2 = fig3.add_subplot(gs3[1])
    im3 = ax2.imshow(pooled_cm, cmap='Blues')
    plt.colorbar(im3, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(['Pred Low','Pred High'])
    ax2.set_yticklabels(['Actual Low','Actual High'])
    for r in range(2):
        for c in range(2):
            ax2.text(c, r, str(pooled_cm[r,c]), ha='center', va='center',
                      fontsize=14, fontweight='bold',
                      color='white' if pooled_cm[r,c]>pooled_cm.max()*0.5
                      else 'black')
    ax2.set_title(f'Pooled CM (sum of all folds)\n'
                   f'AUC={pooled_auc:.3f}  Acc={pooled_acc:.3f}\n'
                   f'Sens={tp/(tp+fn)*100:.1f}%  Spec={tn/(tn+fp)*100:.1f}%')

    ax3 = fig3.add_subplot(gs3[2])
    colors_c = ['#C44E52' if c>0 else '#4C72B0'
                 for c in coef_df['coefficient']]
    ax3.barh(range(len(coef_df)), coef_df['coefficient'],
              color=colors_c, alpha=0.85)
    ax3.set_yticks(range(len(coef_df)))
    ax3.set_yticklabels(coef_df['full_name'], fontsize=8)
    ax3.axvline(0, color='black', lw=1)
    ax3.set_xlabel('Coefficient (features z-scored)')
    ax3.set_title(f'LR Coefficients (full data fit)\n'
                   f'Red(+)=raises P(high arousal)\n'
                   f'Blue(-)=lowers P(high arousal)\n'
                   f'Fit on ALL {len(y)} windows, not a CV fold')

    fig3.suptitle(
        f'Logistic Regression: {label.replace("_"," ")}\n'
        f'C=1.0, L2, balanced, lbfgs — NO hyperparameter tuning\n'
        f'5-fold stratified CV — pooled confusion matrix = sum of all folds',
        fontsize=10, y=1.01)
    plt.tight_layout()
    path3 = os.path.join(output_dir, f'logistic_regression_{label}.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {path3}")
    plt.close()

    return {
        'label': label, 'pooled_auc': pooled_auc,
        'mean_fold_auc': np.mean(fold_aucs), 'auc_sd': np.std(fold_aucs),
        'pooled_acc': pooled_acc, 'f1_pos': pooled_f1p, 'f1_neg': pooled_f1n,
        'n_total': len(y), 'n_loaded': len(df), 'n_dropped': n_dropped,
        'n_high': int((y==1).sum()), 'n_low': int((y==0).sum()),
    }


def print_comparison(results_list):
    print(f"\n{'='*78}")
    print("FINAL MODEL COMPARISON TABLE")
    print(f"{'='*78}")
    print(f"{'Model':<38} {'AUC':>6} {'AUC SD':>7} "
          f"{'Acc':>6} {'F1+':>6} {'F1-':>6}")
    print('-'*78)

    gb = GB_RESULTS.get('all_emotions', {})
    if gb:
        print(f"{'Gradient Boosting (all emotions, ref)':<38} "
              f"{gb['auc']:>6.3f} {gb.get('auc_sd',0):>7.3f} "
              f"{gb['acc']:>6.3f} {gb['f1_pos']:>6.3f} "
              f"{gb['f1_neg']:>6.3f}")

    for r in results_list:
        name = f"Logistic Regression ({r['label'].replace('_',' ')})"
        print(f"{name:<38} {r['pooled_auc']:>6.3f} {r['auc_sd']:>7.3f} "
              f"{r['pooled_acc']:>6.3f} {r['f1_pos']:>6.3f} "
              f"{r['f1_neg']:>6.3f}")

    print(f"{'='*78}")
    print("F1+ = F1 for high arousal class (positive)")
    print("F1- = F1 for low arousal class  (negative)")
    print("AUC SD = standard deviation across 5 CV folds (stability)")


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

    # All emotions
    print("Loading all emotion data...")
    df_all = load_features(args.data_dir, scale=args.scale)
    if df_all is None:
        print("ERROR: No data found. Run [3] PROCESS first.")
        raise SystemExit(1)

    results_all = run_logistic_regression(
        df_all, args.output_dir, 'all_emotions')

    # Fear only
    print("\n\nLoading fear-only data...")
    df_fear = load_features(args.data_dir, scale=args.scale,
                             emotion_filter='fear')

    results_fear = None
    if df_fear is not None and len(df_fear) >= 20:
        results_fear = run_logistic_regression(
            df_fear, args.output_dir, 'fear_only')
    else:
        print("Not enough fear windows for separate analysis (need >= 20).")

    # Final comparison table
    all_results = [results_all]
    if results_fear:
        all_results.append(results_fear)
    print_comparison(all_results)

    print(f"\nOutputs saved to: {args.output_dir}")
    print("Files generated:")
    print("  logistic_regression_all_emotions.png")
    if results_fear:
        print("  logistic_regression_fear_only.png")