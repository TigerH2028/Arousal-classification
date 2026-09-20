"""
FILE: cnn_model.py
PURPOSE: 1D Convolutional Neural Network that learns directly from raw
         EDA and ECG waveforms (two input channels) to predict binary
         arousal state. This is the deep-learning counterpart to your
         existing Gradient Boosting model in emotion_model.py.

REQUIRES: PyTorch
    pip install torch --break-system-packages
    (CPU-only is fine for this dataset size; no GPU required)

USAGE:
    python cnn_model.py --data_path data/cnn_raw/cnn_dataset_60s.npz
                         --output_dir models/cnn_v1
                         --epochs 30
"""

import numpy as np
import json
import os
import argparse
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_global_seed(seed=42):
    """
    Fixes randomness across Python, NumPy, and PyTorch so that two runs
    on the SAME data produce the SAME result. Without this, every run
    starts the CNN's weights from a different random point, which can
    swing AUC by 0.05-0.10+ on small datasets with no change to the data.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Forces deterministic algorithms where available (slightly slower, fully reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Dataset wrapper
# ----------------------------------------------------------------------
class BiosignalDataset(Dataset):
    """
    Wraps EDA + ECG arrays as a 2-channel signal tensor for PyTorch.
    Shape per sample: (2, window_length)  -> channel 0 = EDA, channel 1 = ECG
    """
    def __init__(self, eda, ecg, labels):
        self.eda = eda.astype(np.float32)
        self.ecg = ecg.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.stack([self.eda[idx], self.ecg[idx]], axis=0)  # (2, L)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ----------------------------------------------------------------------
# Model architecture
# ----------------------------------------------------------------------
class BiosignalCNN(nn.Module):
    """
    A compact 1D CNN for two-channel physiological time series.

    Architecture rationale:
      - Three convolutional blocks with increasing channel depth and
        progressively larger receptive fields, so the network can learn
        both fast transients (heartbeats) and slow trends (GSR drift).
      - MaxPooling after each block reduces sequence length, keeping the
        model lightweight enough to train on CPU in a reasonable time.
      - Global average pooling at the end makes the model robust to the
        exact input length, then a small fully-connected head outputs
        a single arousal probability.
    """
    def __init__(self, input_channels=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).squeeze(-1)   # (batch, 128)
        out = self.classifier(x)              # (batch, 1)
        return out.squeeze(-1)


# ----------------------------------------------------------------------
# Single fold training (one train/test split, runs the full epoch loop)
# ----------------------------------------------------------------------
def train_one_fold(eda_train, ecg_train, y_train, eda_test, ecg_test, y_test,
                    device, epochs, batch_size, lr, fold_num, total_folds):
    """
    Trains one CNN from scratch on one fold's train/test split.
    Returns the best-checkpoint predictions on that fold's held-out test set.
    """
    train_ds = BiosignalDataset(eda_train, ecg_train, y_train)
    test_ds = BiosignalDataset(eda_test, ecg_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BiosignalCNN(input_channels=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=3)

    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_probs, all_true = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_true.extend(y_batch.cpu().numpy().tolist())
        val_loss /= len(test_ds)

        try:
            val_auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            val_auc = float('nan')

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"    [Fold {fold_num}/{total_folds}] Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.3f}")

        if val_auc == val_auc and val_auc > best_val_auc:  # val_auc==val_auc excludes NaN
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend(y_batch.numpy().tolist())

    return np.array(all_true), np.array(all_probs)


# ----------------------------------------------------------------------
# 5-fold cross-validated training (matches the Gradient Boosting setup)
# ----------------------------------------------------------------------
def train_cnn_cv(data_path, output_dir, epochs=30, batch_size=16, lr=1e-3,
                  n_folds=5, seed=42):
    """
    Runs proper k-fold stratified cross-validation for the CNN, mirroring
    the 5-fold CV used for the Gradient Boosting model. A fresh CNN is
    trained from scratch on each fold; predictions on each fold's held-out
    windows are pooled to compute one overall AUC/accuracy/F1, and the
    per-fold AUCs are also reported individually so you can see the
    fold-to-fold variance (mean +/- SD) -- this is what makes the CNN
    result directly comparable to the GB model's 5-fold result.
    """
    from sklearn.model_selection import StratifiedKFold

    set_global_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = np.load(data_path, allow_pickle=True)
    eda, ecg, labels = data['eda'], data['ecg'], data['labels']
    print(f"Loaded dataset: {eda.shape[0]} windows, window length {eda.shape[1]} samples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "Only one class present in labels — CNN cannot train. "
            "Re-run cnn_data_loader.py pulling from BOTH positive and "
            "negative emotion folders."
        )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_aucs, fold_accs, fold_f1_pos, fold_f1_neg = [], [], [], []
    all_true_pooled, all_probs_pooled = [], []

    for fold_num, (idx_train, idx_test) in enumerate(skf.split(eda, labels), start=1):
        print(f"\n--- Fold {fold_num}/{n_folds} "
              f"(train={len(idx_train)}, test={len(idx_test)}) ---")

        y_true_fold, y_probs_fold = train_one_fold(
            eda[idx_train], ecg[idx_train], labels[idx_train],
            eda[idx_test], ecg[idx_test], labels[idx_test],
            device, epochs, batch_size, lr, fold_num, n_folds
        )

        fold_preds = (y_probs_fold >= 0.5).astype(int)
        fold_auc = roc_auc_score(y_true_fold, y_probs_fold)
        fold_acc = accuracy_score(y_true_fold, fold_preds)
        fold_f1p = f1_score(y_true_fold, fold_preds, pos_label=1, zero_division=0)
        fold_f1n = f1_score(y_true_fold, fold_preds, pos_label=0, zero_division=0)

        print(f"  Fold {fold_num} result: AUC={fold_auc:.3f}  Acc={fold_acc:.3f}  "
              f"F1+={fold_f1p:.3f}  F1-={fold_f1n:.3f}")

        fold_aucs.append(fold_auc)
        fold_accs.append(fold_acc)
        fold_f1_pos.append(fold_f1p)
        fold_f1_neg.append(fold_f1n)
        all_true_pooled.extend(y_true_fold.tolist())
        all_probs_pooled.extend(y_probs_fold.tolist())

    # Pooled metrics: every window gets exactly one held-out prediction
    # across the 5 folds, so this is the fairest single overall number --
    # directly analogous to how the GB model's CV results are computed.
    pooled_preds = [1 if p >= 0.5 else 0 for p in all_probs_pooled]
    pooled_auc = roc_auc_score(all_true_pooled, all_probs_pooled)
    pooled_acc = accuracy_score(all_true_pooled, pooled_preds)
    pooled_f1_pos = f1_score(all_true_pooled, pooled_preds, pos_label=1)
    pooled_f1_neg = f1_score(all_true_pooled, pooled_preds, pos_label=0)

    results = {
        'model_type': 'CNN_1D_5fold',
        'n_folds': n_folds,
        'pooled_auc': float(pooled_auc),
        'pooled_accuracy': float(pooled_acc),
        'pooled_f1_positive': float(pooled_f1_pos),
        'pooled_f1_negative': float(pooled_f1_neg),
        'fold_auc_mean': float(np.mean(fold_aucs)),
        'fold_auc_sd': float(np.std(fold_aucs)),
        'fold_acc_mean': float(np.mean(fold_accs)),
        'fold_acc_sd': float(np.std(fold_accs)),
        'per_fold_auc': [float(x) for x in fold_aucs],
        'per_fold_accuracy': [float(x) for x in fold_accs],
        'per_fold_f1_positive': [float(x) for x in fold_f1_pos],
        'per_fold_f1_negative': [float(x) for x in fold_f1_neg],
        'n_total_windows': int(len(labels)),
        'epochs_per_fold': epochs,
    }

    with open(os.path.join(output_dir, 'cnn_results_5fold.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FINAL CNN RESULTS — {n_folds}-FOLD CROSS VALIDATION")
    print("=" * 70)
    print(f"  Per-fold AUC:      {[f'{a:.3f}' for a in fold_aucs]}")
    print(f"  Mean AUC:          {np.mean(fold_aucs):.3f}  (SD={np.std(fold_aucs):.3f})")
    print(f"  Pooled AUC:        {pooled_auc:.3f}   <- use this to compare against GB's CV AUC")
    print(f"  Pooled Accuracy:   {pooled_acc:.3f}")
    print(f"  Pooled F1 (pos):   {pooled_f1_pos:.3f}")
    print(f"  Pooled F1 (neg):   {pooled_f1_neg:.3f}")
    print(f"\nSaved results to: {output_dir}/cnn_results_5fold.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                         help="Path to .npz file from cnn_data_loader.py")
    parser.add_argument("--output_dir", default="models/cnn_v1")
    parser.add_argument("--epochs", type=int, default=30,
                         help="Epochs PER FOLD (total training = epochs x n_folds)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_folds", type=int, default=5,
                         help="Number of CV folds. Use 5 to match the Gradient Boosting setup.")
    args = parser.parse_args()

    train_cnn_cv(args.data_path, args.output_dir, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, n_folds=args.n_folds)