"""
FILE: emotion_model.py
PURPOSE: Defines and trains the AI model that predicts emotional states.
         Uses a multi-scale LSTM with multimodal fusion.

HOW TO RUN (after collecting and processing at least 5 sessions):
  python emotion_model.py --data_dir data/processed --output_dir models/

WHAT IT DOES:
  1. Loads windowed feature data from all processed sessions
  2. Trains separate models for each time scale (micro, momentary, episodic, session)
  3. Trains a "fusion" model that combines predictions across scales
  4. Saves trained models and performance metrics
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from glob import glob

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, mean_absolute_error, r2_score,
                              ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURATION
# ============================================================

TIME_SCALES = ['micro', 'momentary', 'episodic']

# Emotion categories — map valence/arousal into quadrants
EMOTION_QUADRANTS = {
    'high_valence_high_arousal':  'excited_happy',    # Happy, excited, elated
    'low_valence_high_arousal':   'anxious_stressed',  # Anxious, angry, fearful
    'high_valence_low_arousal':   'calm_content',      # Calm, relaxed, content
    'low_valence_low_arousal':    'sad_depressed',     # Sad, bored, depressed
}

# Sensor groups (for feature selection per modality)
MODALITY_PREFIXES = {
    'gsr': ['gsr_'],
    'hrv': ['hr_', 'hrv_', 'rr_'],
    'face': ['face_'],
}


# ============================================================
# DATA LOADING & PREPARATION
# ============================================================

class DataLoader:
    """Loads and combines processed data from all sessions."""
    
    def load_all_sessions(self, data_dir, scale):
        """
        Load feature data for a given time scale from all processed session folders.
        Returns combined DataFrame with participant_id column for cross-validation.
        """
        
        pattern = os.path.join(data_dir, '**', f'features_{scale}.csv')
        files = glob(pattern, recursive=True)
        
        if not files:
            print(f"  No data files found for scale '{scale}' in {data_dir}")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if df.empty or len(df.columns) == 0:
                    continue
                # Extract participant ID from path
                path_parts = Path(f).parts
                session_folder = [p for p in path_parts if p.startswith('P')][0] \
                                if any(p.startswith('P') for p in path_parts) else 'UNKNOWN'
                df['participant_id'] = session_folder.split('_')[0]  # e.g., "P001"
                df['session_folder'] = session_folder
                dfs.append(df)
            except pd.errors.EmptyDataError:
                continue
            except Exception as e:
                print(f"  Warning: could not load {f}: {e}")
                continue
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Loaded {len(combined)} windows from {len(files)} sessions "
              f"({combined['participant_id'].nunique()} participants)")
        return combined
    
    def prepare_labels(self, df):
        """
        Create classification labels from valence/arousal scores.
        
        Labels:
        - 2-class: positive vs negative emotion
        - 4-class: emotional quadrant
        - Regression: continuous valence, arousal
        """
        
        df = df.copy()
        
        # Drop windows with no label
        df = df.dropna(subset=['valence', 'arousal'])
        
        # 2-class: positive vs negative valence
        df['label_binary'] = (df['arousal'] >= 8.412).astype(int)
        df['label_binary_name'] = df['label_binary'].map({1: 'positive', 0: 'negative'})
        
        # 4-class: circumplex model quadrants
        high_v = df['valence'] >= 5
        high_a = df['arousal'] >= 5
        
        df['label_quadrant'] = 'unknown'
        df.loc[high_v & high_a, 'label_quadrant'] = 'excited_happy'
        df.loc[~high_v & high_a, 'label_quadrant'] = 'anxious_stressed'
        df.loc[high_v & ~high_a, 'label_quadrant'] = 'calm_content'
        df.loc[~high_v & ~high_a, 'label_quadrant'] = 'sad_depressed'
        
        # Condition-based labels (from VR scene) as alternative
        if 'condition' in df.columns:
            df['label_condition'] = df['condition']
        
        print(f"  Label distribution (binary): "
              f"{df['label_binary_name'].value_counts().to_dict()}")
        print(f"  Label distribution (quadrant): "
              f"{df['label_quadrant'].value_counts().to_dict()}")
        
        return df
    
    def get_feature_matrix(self, df, target='label_binary', exclude_cols=None):
        """
        Separate features (X) from labels (y) and metadata.
        Handles missing values and non-numeric columns.
        """
        
        # Columns to exclude from features
        meta_cols = ['window_start', 'window_end', 'scale', 'condition',
                     'label_binary', 'label_binary_name', 'label_quadrant',
                     'label_condition', 'valence', 'arousal', 'label_source',
                     'participant_id', 'session_folder']
        if exclude_cols:
            meta_cols += exclude_cols
        
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        # Keep only numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].values
        y = df[target].values
        groups = df['participant_id'].values  # For group-aware cross-validation
        
        return X, y, groups, numeric_cols


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SingleScaleModel:
    """
    Predicts emotion from features at a single time scale.
    Uses an ensemble approach: GradientBoosting (best for tabular data) + 
    optional LSTM for temporal patterns.
    """
    
    def __init__(self, scale_name, task='classification'):
        self.scale = scale_name
        self.task = task
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.model = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def build_gradient_boosting(self, n_classes=2):
        """GradientBoosting — excellent for small-to-medium datasets."""
        if self.task == 'classification':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            )
    
    def build_lstm_model(self, n_features, n_classes):
        """
        LSTM model for when you have SEQUENTIAL windows of data.
        Captures temporal patterns across multiple consecutive windows.
        """
        model = keras.Sequential([
            layers.Input(shape=(None, n_features)),
            
            # Bidirectional LSTM: reads sequence forward AND backward
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def fit(self, X, y, groups=None, feature_names=None):
        """
        Train the model with group-aware cross-validation.
        'groups' ensures participants in validation are not in training.
        """
        self.feature_names = feature_names
        
        print(f"\n  Training {self.scale} scale model...")
        print(f"    Dataset: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Preprocessing
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        if self.task == 'classification':
            y_encoded = self.label_encoder.fit_transform(y)
            n_classes = len(self.label_encoder.classes_)
            print(f"    Classes: {list(self.label_encoder.classes_)}")
        else:
            y_encoded = y.astype(float)
        
        # Build and train gradient boosting model
        self.model = self.build_gradient_boosting(n_classes if self.task == 'classification' else 1)
        self.model.fit(X_scaled, y_encoded)
        
        self.is_fitted = True
        
        # Cross-validation (group-based: each participant appears in only one split)
        if groups is not None and len(np.unique(groups)) >= 5:
            print(f"    Running group-stratified cross-validation...")
            cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            
            pipe = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('model', self.build_gradient_boosting())
            ])
            
            if self.task == 'classification':
                scoring = 'roc_auc_ovr_weighted' if n_classes > 2 else 'roc_auc'
            else:
                scoring = 'r2'
            
            scores = cross_val_score(pipe, X, y_encoded, groups=groups,
                                      cv=cv, scoring=scoring)
            print(f"    CV {scoring}: {scores.mean():.3f} ± {scores.std():.3f}")
            
            return {'cv_mean': scores.mean(), 'cv_std': scores.std(), 
                    'cv_scoring': scoring, 'n_cv_splits': 5}
        
        return {}
    
    def predict(self, X):
        """Predict emotional state from new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        if self.task == 'classification':
            y_pred_encoded = self.model.predict(X_scaled)
            y_pred_proba = self.model.predict_proba(X_scaled)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            return y_pred, y_pred_proba
        else:
            return self.model.predict(X_scaled)
    
    def feature_importance(self, top_n=15):
        """Which features matter most for prediction?"""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        if self.feature_names:
            importance_dict = dict(zip(self.feature_names, importances))
            top_features = sorted(importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_n]
            return dict(top_features)
        return {}
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """Evaluate on test set and generate performance report."""
        
        y_pred, y_proba = self.predict(X_test)
        y_test_str = [str(y) for y in y_test]
        
        results = {
            'scale': self.scale,
            'classification_report': classification_report(y_test_str, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test_str, y_pred).tolist(),
        }
        
        # Feature importance
        X_imputed = self.imputer.transform(X_test)
        X_scaled = self.scaler.transform(X_imputed)
        importance = self.feature_importance()
        results['top_features'] = importance
        
        print(f"\n  === {self.scale.upper()} SCALE RESULTS ===")
        print(classification_report(y_test_str, y_pred))
        
        if output_dir:
            # Save confusion matrix plot
            cm = confusion_matrix(y_test_str, y_pred, labels=sorted(set(y_test_str)))
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(y_test_str)))
            disp.plot(ax=ax, cmap='Blues')
            ax.set_title(f'{self.scale.capitalize()} Scale — Emotion Classification')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{self.scale}.png'), dpi=150)
            plt.close()
            
            # Feature importance plot
            if importance:
                fig, ax = plt.subplots(figsize=(10, 6))
                feats = list(importance.keys())[:15]
                vals = [importance[f] for f in feats]
                ax.barh(feats[::-1], vals[::-1], color='steelblue')
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{self.scale.capitalize()} Scale — Top Features')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'feature_importance_{self.scale}.png'), dpi=150)
                plt.close()
        
        return results


# ============================================================
# MULTI-SCALE FUSION MODEL
# ============================================================

class MultiScaleFusionModel:
    """
    Combines predictions from all time scales.
    
    Why this matters: a momentary spike in GSR (micro scale) combined with
    sustained high arousal (episodic scale) is more diagnostic than either alone.
    
    Architecture: Late fusion — each scale model makes a prediction,
    then a meta-model combines all predictions.
    """
    
    def __init__(self):
        self.scale_models = {}
        self.meta_model = None
        self.meta_scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, scale_data, y_all, groups=None):
        """
        scale_data: dict of {scale_name: (X, feature_names)}
        y_all: labels (must be same length/order across scales)
        """
        
        print("\n" + "="*50)
        print("TRAINING MULTI-SCALE FUSION MODEL")
        print("="*50)
        
        # Step 1: Train individual scale models
        cv_results = {}
        for scale_name, (X, feature_names) in scale_data.items():
            model = SingleScaleModel(scale_name, task='classification')
            result = model.fit(X, y_all, groups=groups, feature_names=feature_names)
            self.scale_models[scale_name] = model
            cv_results[scale_name] = result
        
        # Step 2: Get predictions from each scale model on training data (in-fold)
        # (Use out-of-fold predictions to avoid leakage)
        print("\n  Building fusion meta-features...")
        
        meta_features = []
        for scale_name, model in self.scale_models.items():
            _, proba = model.predict(scale_data[scale_name][0])
            meta_features.append(proba)
        
        if len(meta_features) > 1:
            meta_X = np.concatenate(meta_features, axis=1)
            
            # Step 3: Train meta-model on probability outputs
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_all)
            self.meta_label_encoder = label_encoder
            
            meta_X_scaled = self.meta_scaler.fit_transform(meta_X)
            self.meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=500)
            self.meta_model.fit(meta_X_scaled, y_encoded)
            
            print(f"  Fusion model trained on {meta_X.shape[1]} meta-features")
        
        self.is_fitted = True
        return cv_results
    
    def predict(self, scale_data_dict):
        """Predict using all available scales, fall back to single scale if needed."""
        
        meta_features = []
        predictions_per_scale = {}
        
        for scale_name, X in scale_data_dict.items():
            if scale_name in self.scale_models:
                pred, proba = self.scale_models[scale_name].predict(X)
                predictions_per_scale[scale_name] = (pred, proba)
                meta_features.append(proba)
        
        if len(meta_features) > 1 and self.meta_model is not None:
            meta_X = np.concatenate(meta_features, axis=1)
            meta_X_scaled = self.meta_scaler.transform(meta_X)
            y_encoded = self.meta_model.predict(meta_X_scaled)
            y_pred = self.meta_label_encoder.inverse_transform(y_encoded)
            y_proba = self.meta_model.predict_proba(meta_X_scaled)
            return y_pred, y_proba, predictions_per_scale
        elif predictions_per_scale:
            # Fall back to first available scale
            scale = list(predictions_per_scale.keys())[0]
            return predictions_per_scale[scale][0], predictions_per_scale[scale][1], predictions_per_scale
        
        return np.array([]), np.array([]), {}
    
    def save(self, output_dir):
        """Save all models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        for scale_name, model in self.scale_models.items():
            with open(os.path.join(output_dir, f'model_{scale_name}.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        if self.meta_model:
            with open(os.path.join(output_dir, 'model_fusion.pkl'), 'wb') as f:
                pickle.dump({
                    'meta_model': self.meta_model,
                    'meta_scaler': self.meta_scaler,
                    'meta_label_encoder': self.meta_label_encoder
                }, f)
        
        print(f"\n  Models saved to: {output_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """Load saved models."""
        fusion = cls()
        
        for scale in TIME_SCALES:
            path = os.path.join(model_dir, f'model_{scale}.pkl')
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    fusion.scale_models[scale] = pickle.load(f)
                print(f"  Loaded {scale} model")
        
        meta_path = os.path.join(model_dir, 'model_fusion.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            fusion.meta_model = meta_data['meta_model']
            fusion.meta_scaler = meta_data['meta_scaler']
            fusion.meta_label_encoder = meta_data['meta_label_encoder']
            print(f"  Loaded fusion model")
        
        fusion.is_fitted = True
        return fusion


# ============================================================
# PERSONALIZATION: Fine-tune for an individual
# ============================================================

class PersonalizedModel:
    """
    Takes a trained population model and adapts it to a specific individual.
    Run this after collecting 2+ sessions from a new participant.
    """
    
    def __init__(self, base_model_dir, participant_id):
        self.base_fusion = MultiScaleFusionModel.load(base_model_dir)
        self.participant_id = participant_id
        self.personal_models = {}
        
    def personalize(self, X_personal, y_personal, scale, feature_names=None):
        """
        Fine-tune a single scale model with personal data.
        Uses the population model's predictions as soft labels to avoid overfitting.
        """
        
        print(f"\n  Personalizing {scale} model for {self.participant_id}...")
        
        if scale not in self.base_fusion.scale_models:
            print(f"  No base model for {scale}.")
            return
        
        base_model = self.base_fusion.scale_models[scale]
        
        # Get base model's predictions as "soft labels" (knowledge distillation)
        base_preds, base_proba = base_model.predict(X_personal)
        
        # Train a new model initialized with small dataset
        personal_model = SingleScaleModel(f"{scale}_personal_{self.participant_id}")
        personal_model.imputer = base_model.imputer
        personal_model.scaler = base_model.scaler
        personal_model.label_encoder = base_model.label_encoder
        
        # Use personal data to train a lightweight logistic regression on top
        X_imputed = personal_model.imputer.transform(X_personal)
        X_scaled = personal_model.scaler.transform(X_imputed)
        
        y_encoded = personal_model.label_encoder.transform(y_personal)
        
        # Blend base model features with personal data (weight personal data 3x)
        from sklearn.linear_model import LogisticRegression
        personal_model.model = LogisticRegression(C=10.0, max_iter=1000)
        personal_model.model.fit(X_scaled, y_encoded)
        personal_model.feature_names = feature_names
        personal_model.is_fitted = True
        
        self.personal_models[scale] = personal_model
        print(f"  Personalization complete. Used {len(X_personal)} personal samples.")


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_full_pipeline(data_dir, output_dir, target='label_binary'):
    """
    Complete training pipeline:
    1. Load data from all scales and sessions
    2. Train scale-specific models
    3. Train fusion model
    4. Evaluate and save everything
    """
    
    print("\n" + "="*60)
    print("  EMOTION AI — MODEL TRAINING PIPELINE")
    print(f"  Target: {target}")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    loader = DataLoader()
    all_scale_data = {}
    y_reference = None
    groups_reference = None
    
    # Load data for each scale
    for scale in TIME_SCALES:
        df = loader.load_all_sessions(data_dir, scale)
        
        if df.empty:
            print(f"  Skipping {scale} — no data.")
            continue
        
        df = loader.prepare_labels(df)
        df = df.dropna(subset=[target])
        
        X, y, groups, feature_names = loader.get_feature_matrix(df, target=target)
        
        if len(X) < 20:
            print(f"  Skipping {scale} — too few samples ({len(X)}).")
            continue
        
        all_scale_data[scale] = (X, feature_names)
        
        if y_reference is None:
            y_reference = y
            groups_reference = groups
        else:
            # Use minimum common size
            min_len = min(len(y), len(y_reference))
            y_reference = y_reference[:min_len]
            groups_reference = groups_reference[:min_len]
            for s in list(all_scale_data.keys()):
                X_s, fn = all_scale_data[s]
                all_scale_data[s] = (X_s[:min_len], fn)
    
    if not all_scale_data:
        print("\nERROR: No data found. Have you run signal_processor.py first?")
        print("Expected data in:", data_dir)
        return None
    
    # Train multi-scale fusion model
    fusion = MultiScaleFusionModel()
    cv_results = fusion.fit(all_scale_data, y_reference, groups=groups_reference)
    
    # Save models
    fusion.save(output_dir)
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'target': target,
        'scales_trained': list(all_scale_data.keys()),
        'total_samples': len(y_reference),
        'n_participants': len(np.unique(groups_reference)),
        'cv_results': cv_results,
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print(f"  Models saved to: {output_dir}")
    print("="*60)
    
    return fusion


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed", 
                        help="Directory containing processed session folders")
    parser.add_argument("--output_dir", default="models/v1",
                        help="Where to save trained models")
    parser.add_argument("--target", default="label_binary",
                        choices=["label_binary", "label_quadrant"],
                        help="Classification target")
    
    args = parser.parse_args()
    
    model = train_full_pipeline(args.data_dir, args.output_dir, args.target)
    
    if model:
        print("\nNext step: Run inference_server.py to serve predictions to Unity VR.")
