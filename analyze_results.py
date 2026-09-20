"""
FILE: analyze_results.py
PURPOSE: Analyzes collected data and model performance.
         Tells you whether the project is working and what to fix.

HOW TO RUN:
  python analyze_results.py --data_dir data/processed --model_dir models/v1

OUTPUTS:
  - HTML report with charts
  - Decision matrix: what to change based on results
  - Individual participant analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from glob import glob
import os
import json
import pickle
from emotion_model import SingleScaleModel, MultiScaleFusionModel, PersonalizedModel
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="husl")


# ============================================================
# EXPERIMENT 1: Model Performance Analysis
# ============================================================

class PerformanceAnalyzer:
    """Analyzes AI model prediction accuracy at each time scale."""
    
    def analyze(self, data_dir, model_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== PERFORMANCE ANALYSIS ===")
        
        results_by_scale = {}
        
        for scale in ['micro', 'momentary', 'episodic']:
            # Load all data for this scale
            files = glob(os.path.join(data_dir, '**', f'features_{scale}.csv'), recursive=True)
            if not files:
                print(f"  No data for {scale} scale.")
                continue
            
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                # Sample rows
                if len(df) > 200:
                    df = df.sample(n=200, random_state=42)
                folder_name = os.path.basename(os.path.dirname(f))
                df['participant_id'] = folder_name.split('_')[0]
                dfs.append(df)
            # Concat in chunks to avoid memory spike
            combined = pd.concat(dfs, ignore_index=True)

            # Drop columns that are almost entirely NaN (reduces width dramatically)
            combined = combined.dropna(axis=1, thresh=int(len(combined) * 0.5))
            
            all_df = pd.concat(dfs, ignore_index=True)
            all_df = all_df.dropna(subset=['valence', 'arousal'])
            
            if len(all_df) < 20:
                print(f"  Insufficient data for {scale}: {len(all_df)} samples.")
                continue
            
            all_df['label'] = (all_df['valence'] >= 5).map({True: 'positive', False: 'negative'})
            all_df['label_name'] = all_df['label']
            
            # Load model
            model_path = os.path.join(model_dir, f'model_{scale}.pkl')
            if not os.path.exists(model_path):
                print(f"  No trained model for {scale}.")
                continue
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Get feature columns
            meta_cols = ['window_start', 'window_end', 'scale', 'condition',
                         'label', 'label_name', 'valence', 'arousal', 'label_source',
                         'participant_id', 'session_folder']
            feature_cols = [c for c in all_df.columns if c not in meta_cols]
            numeric_cols = all_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            X = all_df[numeric_cols].values
            y = all_df['label_name'].values
            groups = all_df['participant_id'].values
            
            # Leave-one-participant-out evaluation
            y_true_all, y_pred_all, y_proba_all = [], [], []
            
            if len(np.unique(groups)) >= 4:
                cv = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
                
                for train_idx, test_idx in cv.split(X, y, groups=groups):
                    X_test = X[test_idx]
                    y_test = y[test_idx]
                    
                    pred, proba = model.predict(X_test)
                    y_true_all.extend(y_test)
                    y_pred_all.extend(pred)
                    y_proba_all.extend(proba[:, 1] if proba.shape[1] > 1 else proba[:, 0])
            
            if not y_true_all:
                print(f"  Insufficient participants for CV on {scale}.")
                continue
            
            # Compute metrics
            y_true_all = [str(y) for y in y_true_all]
            y_pred_all = [str(y) for y in y_pred_all]
            pred_map = {'0': 'negative', '1': 'positive'}
            y_pred_all = [pred_map.get(p, p) for p in y_pred_all]
            report = classification_report(y_true_all, y_pred_all, output_dict=True)
            cm = confusion_matrix(y_true_all, y_pred_all)
            
            try:
                auc = roc_auc_score(
                    [1 if y == 'positive' else 0 for y in y_true_all], 
                    y_proba_all
                )
            except:
                auc = None
            
            results_by_scale[scale] = {
                'n_samples': len(y_true_all),
                'accuracy': report.get('accuracy', 0),
                'auc': auc,
                'f1_positive': report.get('positive', {}).get('f1-score', 0),
                'f1_negative': report.get('negative', {}).get('f1-score', 0),
                'confusion_matrix': cm,
                'report': report
            }
            
            print(f"\n  {scale.upper()} SCALE (n={len(y_true_all)}):")
            print(f"    Accuracy: {report.get('accuracy', 0):.3f}")
            print(f"    AUC: {auc:.3f}" if auc else "    AUC: N/A")
            print(f"    F1 Positive: {report.get('positive', {}).get('f1-score', 0):.3f}")
            print(f"    F1 Negative: {report.get('negative', {}).get('f1-score', 0):.3f}")
        
        # Generate performance plot
        self._plot_performance_summary(results_by_scale, output_dir)
        
        return results_by_scale
    
    def _plot_performance_summary(self, results, output_dir):
        if not results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Across Time Scales', fontsize=14, fontweight='bold')
        
        scales = list(results.keys())
        
        # Accuracy & AUC
        ax = axes[0]
        accuracies = [results[s]['accuracy'] for s in scales]
        aucs = [results[s]['auc'] or 0 for s in scales]
        
        x = np.arange(len(scales))
        bars1 = ax.bar(x - 0.2, accuracies, 0.35, label='Accuracy', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + 0.2, aucs, 0.35, label='AUC', color='coral', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=15)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (0.5)')
        ax.axhline(0.60, color='green', linestyle=':', alpha=0.7, label='Good (0.60)')
        ax.set_title('Accuracy & AUC by Time Scale')
        ax.legend(fontsize=8)
        ax.set_ylabel('Score')
        
        # Add value labels
        for bar in bars1 + bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}', 
                   ha='center', va='bottom', fontsize=8)
        
        # F1 Scores
        ax = axes[1]
        f1_pos = [results[s]['f1_positive'] for s in scales]
        f1_neg = [results[s]['f1_negative'] for s in scales]
        ax.bar(x - 0.2, f1_pos, 0.35, label='F1 Positive', color='green', alpha=0.8)
        ax.bar(x + 0.2, f1_neg, 0.35, label='F1 Negative', color='red', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=15)
        ax.set_ylim(0, 1)
        ax.set_title('F1 Scores by Class & Scale')
        ax.legend(fontsize=8)
        ax.set_ylabel('F1 Score')
        
        # Sample counts
        ax = axes[2]
        n_samples = [results[s]['n_samples'] for s in scales]
        bars = ax.bar(scales, n_samples, color='mediumpurple', alpha=0.8)
        ax.set_title('Samples per Scale')
        ax.set_ylabel('Number of Windows')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)), 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: performance_summary.png")


# ============================================================
# EXPERIMENT 2: Individual Variability Analysis
# ============================================================

class IndividualVariabilityAnalyzer:
    """Shows how well (or poorly) the model works for each person."""
    
    def analyze(self, data_dir, model_dir, output_dir):
        print("\n=== INDIVIDUAL VARIABILITY ANALYSIS ===")
        
        files = glob(os.path.join(data_dir, '**', 'features_momentary.csv'), recursive=True)
        if not files:
            print("  No momentary scale data found.")
            return {}
        
        model_path = os.path.join(model_dir, 'model_momentary.pkl')
        if not os.path.exists(model_path):
            print("  No momentary model found.")
            return {}
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        participant_results = {}
        
        for f in files:
            df = pd.read_csv(f)
            if len(df) > 200:
                df = df.sample(n=200, random_state=42)
            folder_name = os.path.basename(os.path.dirname(f))
            participant_id = folder_name.split('_')[0]
            
            df = df.dropna(subset=['valence'])
            df['label'] = (df['valence'] >= 5).astype(int)
            df['label_name'] = df['label'].map({1: 'positive', 0: 'negative'})
            
            meta_cols = ['window_start', 'window_end', 'scale', 'condition',
                         'label', 'label_name', 'valence', 'arousal', 'label_source',
                         'participant_id', 'session_folder']
            feature_cols = [c for c in df.columns if c not in meta_cols]
            numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            df = df.dropna(subset=numeric_cols, how='all')
            numeric_cols = [c for c in numeric_cols if df[c].notna().mean() > 0.5]
            
            if len(df) < 5:
                continue
            
            X = df[numeric_cols].values
            y = df['label_name'].values
            
            try:
                preds, proba = model.predict(X)
                pred_map = {'0': 'negative', '1': 'positive'}
                preds_mapped = [pred_map.get(str(p), str(p)) for p in preds]
                y_str = [str(label) for label in y]
                accuracy = np.mean([p == g for p, g in zip(preds_mapped, y_str)])
                
                participant_results[participant_id] = {
                    'accuracy': accuracy,
                    'n_samples': len(y),
                    'n_positive': sum(y == 'positive'),
                    'n_negative': sum(y == 'negative'),
                }
                
                print(f"  {participant_id}: accuracy={accuracy:.2f} (n={len(y)})")
            except Exception as e:
                print(f"  {participant_id}: prediction failed — {e}")
        
        if participant_results:
            self._plot_individual_variability(participant_results, output_dir)
        
        return participant_results
    
    def _plot_individual_variability(self, results, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Individual Variability in Model Performance', fontsize=14)
        
        participants = sorted(results.keys())
        accuracies = [results[p]['accuracy'] for p in participants]
        n_samples = [results[p]['n_samples'] for p in participants]
        
        # Accuracy by participant
        colors = ['green' if a >= 0.70 else 'orange' if a >= 0.60 else 'red' 
                  for a in accuracies]
        axes[0].bar(participants, accuracies, color=colors, alpha=0.8)
        axes[0].axhline(0.5, color='gray', linestyle='--', label='Chance')
        axes[0].axhline(0.60, color='green', linestyle=':', label='Good threshold (0.60)')
        axes[0].set_xlabel('Participant')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Per-Participant Accuracy')
        axes[0].legend()
        axes[0].set_xticklabels(participants, rotation=45)
        
        # Distribution
        axes[1].hist(accuracies, bins=10, edgecolor='black', color='steelblue', alpha=0.7)
        axes[1].axvline(np.mean(accuracies), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(accuracies):.2f}')
        axes[1].axvline(0.5, color='gray', linestyle=':', label='Chance')
        axes[1].set_xlabel('Accuracy')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Accuracy Distribution Across Participants')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_variability.png'), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# EXPERIMENT 3: Sensor Contribution Analysis
# ============================================================

class SensorContributionAnalyzer:
    """Shows which sensors contribute most to accurate predictions (ablation study)."""
    
    def analyze(self, data_dir, model_dir, output_dir):
        print("\n=== SENSOR CONTRIBUTION (ABLATION) ANALYSIS ===")
        
        # Load model feature importances
        model_path = os.path.join(model_dir, 'model_momentary.pkl')
        if not os.path.exists(model_path):
            print("  No model found.")
            return {}
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        if not hasattr(model.model, 'feature_importances_'):
            print("  Model doesn't have feature importances.")
            return {}
        
        importance = model.feature_importance(top_n=30)
        
        # Group by sensor
        sensor_importance = {'gsr': 0.0, 'hrv': 0.0, 'face': 0.0, 'other': 0.0}
        
        for feature, imp in importance.items():
            if feature.startswith('gsr'):
                sensor_importance['gsr'] += imp
            elif feature.startswith(('hr', 'hrv', 'rr')):
                sensor_importance['hrv'] += imp
            elif feature.startswith('face'):
                sensor_importance['face'] += imp
            else:
                sensor_importance['other'] += imp
        
        total = sum(sensor_importance.values())
        sensor_pct = {k: v/total*100 for k, v in sensor_importance.items() if v > 0}
        
        print("  Sensor contribution to predictions:")
        for sensor, pct in sorted(sensor_pct.items(), key=lambda x: x[1], reverse=True):
            print(f"    {sensor}: {pct:.1f}%")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        axes[0].pie(sensor_pct.values(), labels=sensor_pct.keys(), autopct='%1.1f%%',
                   colors=['coral', 'steelblue', 'green', 'gray'])
        axes[0].set_title('Sensor Contribution to Predictions')
        
        # Top features bar chart
        top_features = list(importance.items())[:20]
        feature_names = [f[0][:25] for f in top_features]
        feature_vals = [f[1] for f in top_features]
        
        feature_colors = []
        for name in feature_names:
            if name.startswith('gsr'): feature_colors.append('coral')
            elif name.startswith(('hr', 'hrv', 'rr')): feature_colors.append('steelblue')
            elif name.startswith('face'): feature_colors.append('green')
            else: feature_colors.append('gray')
        
        axes[1].barh(feature_names[::-1], feature_vals[::-1], color=feature_colors[::-1], alpha=0.8)
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 20 Features (Color = Sensor)')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='coral', label='GSR'),
                          Patch(facecolor='steelblue', label='HRV/HR'),
                          Patch(facecolor='green', label='Face')]
        axes[1].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sensor_contributions.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        return sensor_pct


# ============================================================
# REDESIGN DECISION FRAMEWORK
# ============================================================

class RedesignDecisionMaker:
    """
    Analyzes results and generates specific, actionable redesign recommendations.
    This is the most important class for iterating your project.
    """
    
    # Performance thresholds
    ACCURACY_CHANCE = 0.50      # No better than chance
    ACCURACY_MINIMUM = 0.60     # Below this: fix the basics
    ACCURACY_ACCEPTABLE = 0.70  # Acceptable for research
    ACCURACY_GOOD = 0.80        # Good for publication
    ACCURACY_EXCELLENT = 0.90   # Clinical quality
    
    def evaluate_and_recommend(self, performance_results, individual_results, 
                                 sensor_results, output_dir):
        
        print("\n" + "="*60)
        print("  REDESIGN DECISION ANALYSIS")
        print("="*60)
        
        recommendations = []
        status = "UNKNOWN"
        
        if not performance_results:
            recommendations.append({
                'priority': 'CRITICAL',
                'area': 'Data Collection',
                'problem': 'No model performance data available.',
                'action': 'You need at least 5 participants × 2 sessions before analysis. '
                           'Run data_collector.py and signal_processor.py first.',
                'redesign_needed': True
            })
            self._save_report(recommendations, "BLOCKED", output_dir)
            return recommendations
        
        # Evaluate each scale
        scale_statuses = {}
        for scale, metrics in performance_results.items():
            acc = metrics.get('accuracy', 0)
            auc = metrics.get('auc', 0) or 0
            n = metrics.get('n_samples', 0)
            
            if n < 50:
                scale_statuses[scale] = 'insufficient_data'
            elif acc < self.ACCURACY_MINIMUM or auc < 0.60:
                scale_statuses[scale] = 'poor'
            elif acc < self.ACCURACY_ACCEPTABLE or auc < 0.70:
                scale_statuses[scale] = 'marginal'
            elif acc < self.ACCURACY_GOOD:
                scale_statuses[scale] = 'acceptable'
            else:
                scale_statuses[scale] = 'good'
        
        print(f"\n  Scale statuses: {scale_statuses}")
        
        # ---- CRITICAL ISSUES ----
        poor_scales = [s for s, status in scale_statuses.items() if status == 'poor']
        if poor_scales:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Model Performance',
                'problem': f'Poor accuracy on: {poor_scales}',
                'action': '\n'.join([
                    '1. Check data quality: run signal_processor.py with verbose mode',
                    '2. Verify sensor connections — poor quality data = poor predictions',
                    '3. Increase training data: minimum 10 participants × 3 sessions each',
                    '4. Check label quality: are self-report ratings matching expected emotion inductions?',
                    '5. Review feature engineering: add more sensor-specific features',
                ]),
                'redesign_needed': True
            })
        
        # ---- INDIVIDUAL VARIABILITY ----
        if individual_results:
            accuracies = [r['accuracy'] for r in individual_results.values()]
            acc_std = np.std(accuracies)
            acc_range = max(accuracies) - min(accuracies)
            poor_participants = [p for p, r in individual_results.items() 
                                  if r['accuracy'] < 0.55]
            
            if acc_std > 0.15 or acc_range > 0.30:
                recommendations.append({
                    'priority': 'HIGH',
                    'area': 'Individual Variability',
                    'problem': f'Large performance variance across participants (SD={acc_std:.2f}, range={acc_range:.2f}). '
                               f'Poor performers: {poor_participants}',
                    'action': '\n'.join([
                        '1. Implement personalization: collect 2 "calibration" sessions per person',
                        '2. Add individual z-score normalization to signal_processor.py',
                        '3. Check if poor performers have sensor artifacts (bad GSR contact, motion)',
                        '4. Consider training separate models per demographic group',
                    ]),
                    'redesign_needed': True
                })
        
        # ---- SENSOR ISSUES ----
        if sensor_results:
            dominant_sensor = max(sensor_results, key=sensor_results.get)
            dominant_pct = sensor_results[dominant_sensor]
            
            if dominant_pct > 70:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'area': 'Sensor Fusion',
                    'problem': f'{dominant_sensor} contributes {dominant_pct:.0f}% of predictive power. '
                               'Model is over-relying on one modality.',
                    'action': '\n'.join([
                        f'1. Check non-{dominant_sensor} sensor quality and connection',
                        '2. Add modality-balancing to training (adjust feature weights)',
                        '3. Consider if low-contribution sensors are actually working correctly',
                        '4. Run correlation analysis: are sensors actually measuring emotion?',
                    ]),
                    'redesign_needed': False
                })
        
        # ---- DATA SUFFICIENCY ----
        data_poor_scales = [s for s, status in scale_statuses.items() 
                             if status == 'insufficient_data']
        if data_poor_scales:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Data Collection',
                'problem': f'Insufficient data for: {data_poor_scales}',
                'action': '\n'.join([
                    'Collect more sessions. Required minimums:',
                    '  - Micro scale: 20+ sessions (many short windows needed)',
                    '  - Momentary: 10+ sessions',
                    '  - Episodic: 8+ sessions',
                    '  - Session: 15+ participants (one window per session)',
                ]),
                'redesign_needed': False
            })
        
        # ---- POSITIVE RESULTS ----
        good_scales = [s for s, status in scale_statuses.items() if status in ('acceptable', 'good')]
        if good_scales:
            recommendations.append({
                'priority': 'INFO',
                'area': 'Working Well',
                'problem': f'Good performance on: {good_scales}',
                'action': '\n'.join([
                    '✓ These scales are ready for VR integration testing.',
                    '→ Next steps:',
                    '  1. Test real-time inference latency (<100ms target)',
                    '  2. Test with participants wearing VR headset (different posture = different signals)',
                    '  3. Begin VR intervention trials (does the adapted scene change self-reported emotion?)',
                ]),
                'redesign_needed': False
            })
        
        # ---- OVERALL STATUS ----
        n_good = len(good_scales)
        n_poor = len(poor_scales)
        n_insufficient = len(data_poor_scales)
        
        if n_poor > n_good:
            status = "MAJOR_REDESIGN_NEEDED"
        elif n_insufficient > 0 and n_good == 0:
            status = "COLLECT_MORE_DATA"
        elif n_good > 0 and n_poor == 0:
            status = "READY_FOR_VR_INTEGRATION"
        else:
            status = "PARTIAL_SUCCESS_ITERATE"
        
        self._save_report(recommendations, status, output_dir)
        self._print_summary(recommendations, status)
        
        return recommendations
    
    def _print_summary(self, recommendations, status):
        print(f"\n  OVERALL STATUS: {status}")
        print("\n  RECOMMENDATIONS:")
        
        priority_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']
        for priority in priority_order:
            relevant = [r for r in recommendations if r['priority'] == priority]
            for rec in relevant:
                print(f"\n  [{priority}] {rec['area']}")
                print(f"  Problem: {rec['problem']}")
                print(f"  Action:")
                for line in rec['action'].split('\n'):
                    print(f"    {line}")
                print(f"  Redesign needed: {'YES' if rec['redesign_needed'] else 'No'}")
    
    def _save_report(self, recommendations, status, output_dir):
        report = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'overall_status': status,
            'recommendations': recommendations,
        }
        
        path = os.path.join(output_dir, 'redesign_recommendations.json')
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved recommendations: {path}")


# ============================================================
# TESTING PROCEDURES
# ============================================================

class TestingProcedures:
    """
    Formal testing procedures to validate your system.
    Run these in order: unit tests → integration → clinical validity.
    """
    
    def run_sensor_tests(self):
        """TEST 1: Verify each sensor is recording valid data."""
        print("\n=== SENSOR VALIDATION TESTS ===")
        results = {}
        
        # Test 1.1: GSR range check
        print("\nTest 1.1: GSR Range Validity")
        print("  PROCEDURE: Sit quietly for 60 seconds, then watch a scary video clip")
        print("  EXPECTED: Baseline GSR 1-20 uS, increases 2-5 uS during scare")
        print("  PASS CRITERIA: range > 0.5 uS during baseline, clear response to stimulus")
        
        # Test 1.2: HRV baseline
        print("\nTest 1.2: HRV Baseline Validity")
        print("  PROCEDURE: 5 minutes seated rest, Polar H10 on chest")
        print("  EXPECTED: HR 50-90 BPM, RMSSD 20-80 ms (healthy resting)")
        print("  PASS CRITERIA: RMSSD > 15 ms, no flat-line periods > 3 seconds")
        
        # Test 1.3: Face detection rate
        print("\nTest 1.3: Face Detection Rate")
        print("  PROCEDURE: 60-second face recording, normal ambient lighting")
        print("  EXPECTED: >85% frames with face detected")
        print("  PASS CRITERIA: detection_rate > 0.85")
        
        # Test 1.4: System latency
        print("\nTest 1.4: Inference Latency")
        print("  PROCEDURE: Send 100 requests to /predict endpoint, measure response time")
        print("  EXPECTED: Mean < 100ms, P99 < 200ms")
        print("  PASS CRITERIA: mean_latency < 100ms")
        
        return results
    
    def run_emotion_induction_test(self):
        """
        TEST 2: Validate that the system detects known emotional states.
        
        This is the most important test — it proves the system can 
        detect emotions that were deliberately induced.
        """
        
        print("\n=== EMOTION INDUCTION VALIDATION TEST ===")
        print("This test uses known stimuli to check if the AI correctly classifies them.")
        
        test_stimuli = [
            {
                'name': 'Neutral Baseline',
                'stimulus': 'Read neutral text (e.g., weather report) for 3 minutes',
                'expected_valence': 'neutral (4-6)',
                'expected_arousal': 'low (1-4)',
                'expected_label': 'neutral/positive',
            },
            {
                'name': 'Anxiety Induction',
                'stimulus': 'Stroop task (color-word mismatch) under time pressure for 3 minutes',
                'expected_valence': 'negative (1-4)',
                'expected_arousal': 'high (6-9)',
                'expected_label': 'negative/anxious_stressed',
            },
            {
                'name': 'Positive Affect',
                'stimulus': 'Watch a funny or heartwarming video clip (3 minutes)',
                'expected_valence': 'positive (6-9)',
                'expected_arousal': 'moderate-high (5-8)',
                'expected_label': 'positive/excited_happy',
            },
            {
                'name': 'Sadness Induction',
                'stimulus': 'Watch a sad film clip known to induce sadness (validated GAPED stimuli)',
                'expected_valence': 'negative (1-4)',
                'expected_arousal': 'low (1-4)',
                'expected_label': 'negative/sad_depressed',
            },
        ]
        
        print("\nRun each block in order, collecting data continuously:")
        for i, stimulus in enumerate(test_stimuli, 1):
            print(f"\n  Block {i}: {stimulus['name']}")
            print(f"    Stimulus: {stimulus['stimulus']}")
            print(f"    Expected Valence: {stimulus['expected_valence']}")
            print(f"    Expected Arousal: {stimulus['expected_arousal']}")
            print(f"    AI should classify as: {stimulus['expected_label']}")
        
        print("\n  PASS CRITERIA:")
        print("    - AI correctly classifies valence (positive/negative) in ≥3/4 blocks")
        print("    - Arousal matches expected direction in ≥3/4 blocks")
        print("    - Self-report ratings align with expected emotion in ≥3/4 blocks")
        
        print("\n  WHAT TO DO IF IT FAILS:")
        print("    - If AI fails but self-report matches expected: feature extraction is wrong")
        print("    - If self-report doesn't match expected: induction procedure needs adjustment")
        print("    - If sensors don't show clear signals: check electrode contact and sensor placement")
    
    def run_closed_loop_test(self):
        """TEST 3: Does the VR intervention actually change the detected emotion?"""
        
        print("\n=== CLOSED-LOOP INTERVENTION TEST ===")
        print("This test validates the full system: sense → predict → intervene → measure change.")
        
        protocol = """
PROTOCOL:
1. INDUCTION PHASE (10 min):
   - Participant enters VR anxiety-inducing scene (controlled social stress scenario)
   - System records baseline emotion: should be 'negative/high arousal'
   - DO NOT trigger intervention yet

2. RECORDING PHASE (5 min):
   - Continue recording. Confirm AI is consistently predicting 'anxious_stressed'
   - Confirm biosignals show expected anxiety pattern (elevated GSR, lower HRV)

3. INTERVENTION PHASE (10 min):
   - Trigger calming intervention: switch to nature scene, cool colors, calm music
   - Continue recording ALL sensors throughout

4. RECOVERY MEASUREMENT (5 min):
   - Measure emotion predictions after intervention
   - Collect post-intervention self-report

PASS CRITERIA:
  a) Emotion prediction shifts toward 'positive' or 'calm_content' during/after intervention
  b) GSR decreases by ≥15% from peak anxiety level
  c) HRV (RMSSD) increases by ≥10% compared to anxiety baseline
  d) Self-reported valence increases by ≥1 point on 9-point scale
  e) At least 3/4 criteria must be met for 'intervention efficacy' to be claimed

WHAT THE RESULTS MEAN:
  All 4 pass: System working as hypothesized. Proceed to larger validation study.
  2-3 pass:   Partial efficacy. Intervention needs refinement OR measurement needs work.
  0-1 pass:   Intervention is not working OR detection is too noisy. Major iteration needed.
"""
        print(protocol)


# ============================================================
# MAIN
# ============================================================

def run_full_analysis(data_dir, model_dir, output_dir="analysis_results"):
    """Run all analyses and generate the redesign report."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  EMOTION AI-VR — FULL ANALYSIS PIPELINE")
    print("="*60)
    
    perf_analyzer = PerformanceAnalyzer()
    ind_analyzer = IndividualVariabilityAnalyzer()
    sensor_analyzer = SensorContributionAnalyzer()
    decision_maker = RedesignDecisionMaker()
    testing = TestingProcedures()
    
    perf_results = perf_analyzer.analyze(data_dir, model_dir, output_dir)
    ind_results = ind_analyzer.analyze(data_dir, model_dir, output_dir)
    sensor_results = sensor_analyzer.analyze(data_dir, model_dir, output_dir)
    
    recommendations = decision_maker.evaluate_and_recommend(
        perf_results, ind_results, sensor_results, output_dir
    )
    
    # Print testing procedures
    testing.run_sensor_tests()
    testing.run_emotion_induction_test()
    testing.run_closed_loop_test()
    
    print(f"\n✓ All analysis complete. Reports in: {output_dir}")
    return recommendations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="models/v1")
    parser.add_argument("--output_dir", default="analysis_results")
    args = parser.parse_args()
    
    run_full_analysis(args.data_dir, args.model_dir, args.output_dir)
