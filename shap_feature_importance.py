'''
FILE: shap_feature_importance.py
PURPOSE: Computes and visualizes SHAP feature importances across three tuned classifiers 
         (LR-L2, Gradient Boosting, SVM-RBF) on a unified log-odds scale. Generates 
         model-specific All vs. Fear comparison plots (separate per model and panel), 
         individual feature breakdowns, and outputs Top 10 summary tables per model.

USAGE:
    python shap_feature_importance.py --data_dir data/processed --output_dir analysis_results
'''

import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

REAL_SENSOR_FEATURES = [
    'gsr_min', 'gsr_max', 'gsr_mean', 'gsr_std', 'gsr_range',
    'gsr_slope', 'gsr_scl_mean', 'gsr_scr_amplitude', 'gsr_peak_count',
    'hr_mean', 'hr_max', 'hr_min', 'hr_std', 'rr_mean', 'hrv_pnn50',
]

FEATURE_FULLNAMES = {
    'gsr_min': 'GSR Min', 'gsr_max': 'GSR Max', 'gsr_mean': 'GSR Mean',
    'gsr_std': 'GSR SD', 'gsr_range': 'GSR Range', 'gsr_slope': 'GSR Slope',
    'gsr_scl_mean': 'SCL Mean', 'gsr_scr_amplitude': 'SCR Ampl.',
    'gsr_peak_count': 'Peak Count', 'hr_mean': 'HR Mean', 'hr_max': 'HR Max',
    'hr_min': 'HR Min', 'hr_std': 'HR SD', 'rr_mean': 'RR Mean', 'hrv_pnn50': 'pNN50',
}


def load_features(data_dir, scale='episodic', emotion_filter=None):
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
            for key in ['anger', 'fear', 'sadness', 'disgust', 'amusement', 'gratitude', 'neutral', 'tenderness']:
                if key in folder.lower():
                    emotion = key
                    break
            df['emotion'] = emotion

            if emotion_filter and emotion != emotion_filter:
                continue

            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None, None

    combined = pd.concat(dfs, ignore_index=True)
    combined['arousal_binary'] = (combined['arousal'] >= 5.0).astype(int)
    
    avail_features = [f for f in REAL_SENSOR_FEATURES if f in combined.columns]
    X = combined[avail_features].dropna()
    y = combined.loc[X.index, 'arousal_binary']

    return X, y


def compute_shap_importances(X, y, feature_names, best_params_dict):
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X, columns=feature_names)

    results = {}

    # 1. Gradient Boosting (TreeExplainer - Margin/Log-Odds space)
    gb_params = best_params_dict.get('GradBoost', {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1})
    gb_model = GradientBoostingClassifier(**gb_params, random_state=42)
    gb_model.fit(X_df, y)

    gb_explainer = shap.TreeExplainer(gb_model)
    gb_shap_vals = gb_explainer(X_df)
    gb_vals = gb_shap_vals.values[:, :, 1] if len(gb_shap_vals.shape) == 3 else gb_shap_vals.values
    gb_importance = np.abs(gb_vals).mean(axis=0)

    results['GradBoost'] = {
        'model': gb_model,
        'explainer': gb_explainer,
        'shap_values': gb_vals,
        'importance': pd.Series(gb_importance, index=feature_names)
    }

    # 2. Logistic Regression L2 (LinearExplainer - Log-Odds space)
    lr_params = best_params_dict.get('LR-L2', {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'})
    lr_model = LogisticRegression(**lr_params, random_state=42)
    lr_model.fit(X_df, y)

    lr_explainer = shap.LinearExplainer(lr_model, X_df)
    lr_shap_vals = lr_explainer.shap_values(X_df)
    lr_vals = lr_shap_vals[1] if isinstance(lr_shap_vals, list) else lr_shap_vals
    lr_importance = np.abs(lr_vals).mean(axis=0)

    results['LR-L2'] = {
        'model': lr_model,
        'explainer': lr_explainer,
        'shap_values': lr_vals,
        'importance': pd.Series(lr_importance, index=feature_names)
    }

    # 3. SVM RBF Kernel (KernelExplainer - decision_function for Log-Odds alignment)
    svm_params = best_params_dict.get('SVM', {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'})
    svm_model = SVC(**svm_params, probability=True, random_state=42)
    svm_model.fit(X_df, y)

    background_summary = shap.kmeans(X_df, min(15, len(X_df)))
    svm_explainer = shap.KernelExplainer(svm_model.decision_function, background_summary)
    svm_vals = svm_explainer.shap_values(X_df, nsamples=min(100, len(X_df)))
    svm_importance = np.abs(svm_vals).mean(axis=0)

    results['SVM'] = {
        'model': svm_model,
        'explainer': svm_explainer,
        'shap_values': svm_vals,
        'importance': pd.Series(svm_importance, index=feature_names)
    }

    return results, X_df


def plot_model_specific_comparisons(results_all, results_fear, output_dir):
    models = ['GradBoost', 'LR-L2', 'SVM']
    palette = ['#C44E52', '#4C72B0']  # Red: All Emotions, Blue: Fear Only
    
    # 1. Save individual plot per model
    for model in models:
        df_all = results_all[model]['importance'].reset_index()
        df_all.columns = ['Feature', 'Mean_Abs_SHAP']
        df_all['Context'] = 'All Emotions'

        df_fear = results_fear[model]['importance'].reset_index()
        df_fear.columns = ['Feature', 'Mean_Abs_SHAP']
        df_fear['Context'] = 'Fear Only'

        combined = pd.concat([df_all, df_fear], axis=0)
        combined['Full_Name'] = combined['Feature'].map(FEATURE_FULLNAMES)

        order = df_fear.sort_values(by='Mean_Abs_SHAP', ascending=False)['Feature'].tolist()
        combined['Feature_Cat'] = pd.Categorical(combined['Feature'], categories=order, ordered=True)
        combined = combined.sort_values('Feature_Cat')

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        sns.barplot(
            data=combined,
            x='Mean_Abs_SHAP',
            y='Full_Name',
            hue='Context',
            palette=palette,
            ax=ax
        )

        ax.set_title(f'Feature Importance: All Emotions vs. Fear Only ({model})', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Mean |SHAP Value| (Log-Odds Impact)', fontsize=11)
        ax.set_ylabel('Physiological Features', fontsize=11)
        ax.legend(title='Context', loc='lower right', frameon=True)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'shap_comparison_all_vs_fear_{model.lower().replace("-", "_")}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()

    # 2. Side-by-side 1x3 panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=300, sharey=False)
    for idx, model in enumerate(models):
        df_all = results_all[model]['importance'].reset_index()
        df_all.columns = ['Feature', 'Mean_Abs_SHAP']
        df_all['Context'] = 'All Emotions'

        df_fear = results_fear[model]['importance'].reset_index()
        df_fear.columns = ['Feature', 'Mean_Abs_SHAP']
        df_fear['Context'] = 'Fear Only'

        combined = pd.concat([df_all, df_fear], axis=0)
        combined['Full_Name'] = combined['Feature'].map(FEATURE_FULLNAMES)
        
        order = df_fear.sort_values(by='Mean_Abs_SHAP', ascending=False)['Feature'].tolist()
        combined['Feature_Cat'] = pd.Categorical(combined['Feature'], categories=order, ordered=True)
        combined = combined.sort_values('Feature_Cat')

        sns.barplot(
            data=combined,
            x='Mean_Abs_SHAP',
            y='Full_Name',
            hue='Context',
            palette=palette,
            ax=axes[idx]
        )
        axes[idx].set_title(f'{model}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Mean |SHAP Value|', fontsize=10)
        axes[idx].set_ylabel('Physiological Features' if idx == 0 else '', fontsize=11)

    plt.suptitle('Feature Importance Comparison: All Emotions vs. Fear Only across Models', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    panel_path = os.path.join(output_dir, 'shap_comparison_all_vs_fear_all_models_panel.png')
    plt.savefig(panel_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_feature_shap_breakdown(results_all, results_fear, output_dir):
    feature_dir = os.path.join(output_dir, 'feature_shap_breakdowns')
    os.makedirs(feature_dir, exist_ok=True)

    features = list(results_all['GradBoost']['importance'].index)

    for feat in features:
        data = []
        for model in ['GradBoost', 'LR-L2', 'SVM']:
            data.append({
                'Model': model,
                'Context': 'All Emotions',
                'SHAP': results_all[model]['importance'][feat]
            })
            data.append({
                'Model': model,
                'Context': 'Fear Only',
                'SHAP': results_fear[model]['importance'][feat]
            })
        
        df_feat = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        sns.barplot(
            data=df_feat,
            x='Model',
            y='SHAP',
            hue='Context',
            palette=['#C44E52', '#4C72B0'],
            ax=ax
        )
        fname = FEATURE_FULLNAMES.get(feat, feat)
        ax.set_title(f'SHAP Impact Breakdown — {fname} ({feat})', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean |SHAP Value| (Log-Odds)', fontsize=10)
        ax.set_xlabel('Model Classifier', fontsize=10)
        ax.legend(loc='upper right')
        plt.tight_layout()

        plt.savefig(os.path.join(feature_dir, f'shap_breakdown_{feat}.png'), dpi=200)
        plt.close()


def plot_shap_summary_beeswarm(results, X_df, output_dir, label_name="Dataset", max_display=15):
    for model_name in ['GradBoost', 'LR-L2', 'SVM']:
        res = results[model_name]
        plt.figure(figsize=(9, 8), dpi=300)
        shap_vals = res['shap_values']
        
        shap.summary_plot(
            shap_vals, 
            X_df, 
            show=False, 
            max_display=max_display, 
            plot_type="dot"
        )
            
        plt.title(f'SHAP Directionality Summary: {model_name} ({label_name})\n(Log-Odds Scale)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        output_filename = os.path.join(
            output_dir, 
            f"shap_beeswarm_{model_name.lower().replace('-', '_')}_{label_name.lower().replace(' ', '_')}_all15.png"
        )
        plt.savefig(output_filename, dpi=300)
        plt.close()


def generate_top10_table(results_all, results_fear):
    top10_rows = []
    models = ['GradBoost', 'LR-L2', 'SVM']

    for rank in range(10):
        row = {'Rank': rank + 1}
        for model in models:
            s_all = results_all[model]['importance'].sort_values(ascending=False)
            f_all = s_all.index[rank]
            v_all = s_all.iloc[rank]
            row[f'{model} (All)'] = f"{FEATURE_FULLNAMES.get(f_all, f_all)} ({v_all:.3f})"

            s_fear = results_fear[model]['importance'].sort_values(ascending=False)
            f_fear = s_fear.index[rank]
            v_fear = s_fear.iloc[rank]
            row[f'{model} (Fear)'] = f"{FEATURE_FULLNAMES.get(f_fear, f_fear)} ({v_fear:.3f})"

        top10_rows.append(row)

    return pd.DataFrame(top10_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--output_dir', default='analysis_results')
    parser.add_argument('--scale', default='episodic', choices=['micro', 'momentary', 'episodic'])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    best_params = {
        'GradBoost': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8},
        'LR-L2': {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'},
        'SVM': {'C': 100.0, 'gamma': 'scale', 'kernel': 'rbf'}
    }

    X_all, y_all = load_features(args.data_dir, scale=args.scale)
    scaler = StandardScaler()
    X_scaled_all = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)
    results_all, X_proc_all = compute_shap_importances(X_scaled_all, y_all, list(X_all.columns), best_params)

    X_fear, y_fear = load_features(args.data_dir, scale=args.scale, emotion_filter='fear')
    scaler_f = StandardScaler()
    X_scaled_fear = pd.DataFrame(scaler_f.fit_transform(X_fear), columns=X_fear.columns)
    results_fear, X_proc_fear = compute_shap_importances(X_scaled_fear, y_fear, list(X_fear.columns), best_params)

    plot_model_specific_comparisons(results_all, results_fear, args.output_dir)
    plot_per_feature_shap_breakdown(results_all, results_fear, args.output_dir)
    plot_shap_summary_beeswarm(results_all, X_proc_all, args.output_dir, label_name="All Emotions")
    plot_shap_summary_beeswarm(results_fear, X_proc_fear, args.output_dir, label_name="Fear Only")

    top10_df = generate_top10_table(results_all, results_fear)
    top10_csv = os.path.join(args.output_dir, 'top10_shap_features_all_vs_fear.csv')
    top10_df.to_csv(top10_csv, index=False)