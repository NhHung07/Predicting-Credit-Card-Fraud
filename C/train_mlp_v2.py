#!/usr/bin/env python3
"""
MLP C v2 - CPU-friendly Residual MLP with Entity Embeddings
Project: Predicting Credit Card Fraud (Kaggle Playground Series S6E11)
Member: C (Pham Hong Nam)
Experiment: mlp_c_v2

Key improvements over the inherited baseline:
- Fold-local categorical encoding and numerical scaling (prevents preprocessing leakage).
- Unknown/missing categories use embedding index 0 (never a negative embedding index).
- Residual MLP blocks.
- Early stopping and restoration of the best validation-AUC checkpoint.
- Test predictions use the same best checkpoint selected for each fold.
- CPU-safe DataLoader settings (pin_memory=False).
"""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "mlp_c_v2"
MEMBER = "C"
SEED = 42
N_SPLITS = 5

# CPU-friendly starting configuration. Tune only after getting a baseline.
EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 4096
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
HIDDEN_DIM = 256
EMBEDDING_DIM_CAP = 16

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "data" / "folds.csv"
SAMPLE_SUB_PATH = ROOT / "data" / "sample_submission.csv"

OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"

CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

TARGET_COL = "is_fraud"
ID_COL = "id"

# Limit CPU thread usage by default so the machine remains responsive.
# Override in PowerShell if needed: $env:MLP_NUM_THREADS= "6"
CPU_THREADS = int(os.getenv("MLP_NUM_THREADS", "4"))
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class ResidualTabularMLP(nn.Module):
    def __init__(self, cardinalities, num_continuous: int):
        super().__init__()

        self.embeddings = nn.ModuleList()
        total_embedding_dim = 0

        for cardinality in cardinalities:
            # Index 0 is reserved for unknown/missing categories.
            emb_dim = min(
                EMBEDDING_DIM_CAP,
                max(4, int(round(cardinality ** 0.5)) + 1),
            )
            self.embeddings.append(nn.Embedding(cardinality, emb_dim))
            total_embedding_dim += emb_dim

        input_dim = total_embedding_dim + num_continuous

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(HIDDEN_DIM, DROPOUT),
            ResidualBlock(HIDDEN_DIM, DROPOUT),
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x_cat, x_cont):
        embedding_outputs = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat(embedding_outputs + [x_cont], dim=1)
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------
# Fold-local preprocessing
# ---------------------------------------------------------------------
def encode_categories(train_part, other_parts, cat_cols):
    """Fit category maps on this fold's training rows only. Unknown = 0."""
    encoded_train = np.zeros((len(train_part), len(cat_cols)), dtype=np.int64)
    encoded_others = [
        np.zeros((len(part), len(cat_cols)), dtype=np.int64)
        for part in other_parts
    ]
    cardinalities = []

    for col_idx, col in enumerate(cat_cols):
        train_values = train_part[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(train_values.unique())
        mapping = {value: i + 1 for i, value in enumerate(categories)}
        cardinality = len(mapping) + 1  # include unknown/missing index 0
        cardinalities.append(cardinality)

        encoded_train[:, col_idx] = (
            train_values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
        )

        for part_idx, part in enumerate(other_parts):
            values = part[col].astype("string").fillna("__MISSING__")
            encoded_others[part_idx][:, col_idx] = (
                values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
            )

    return encoded_train, encoded_others, cardinalities


def prepare_continuous(train_part, other_parts, cont_cols):
    """Fit imputation values and scaler on this fold's training rows only."""
    x_train_df = train_part[cont_cols].copy()
    x_other_dfs = [part[cont_cols].copy() for part in other_parts]

    for col in cont_cols:
        x_train_df[col] = pd.to_numeric(x_train_df[col], errors="coerce")
        x_train_df[col] = x_train_df[col].replace([np.inf, -np.inf], np.nan)
        median = x_train_df[col].median()
        if pd.isna(median):
            median = 0.0
        x_train_df[col] = x_train_df[col].fillna(median)

        for other_df in x_other_dfs:
            other_df[col] = pd.to_numeric(other_df[col], errors="coerce")
            other_df[col] = other_df[col].replace([np.inf, -np.inf], np.nan)
            other_df[col] = other_df[col].fillna(median)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df).astype(np.float32)
    x_others = [
        scaler.transform(other_df).astype(np.float32)
        for other_df in x_other_dfs
    ]
    return x_train, x_others


# ---------------------------------------------------------------------
# Training / inference helpers
# ---------------------------------------------------------------------
def predict_probabilities(model, x_cat, x_cont):
    model.eval()
    with torch.no_grad():
        logits = model(x_cat, x_cont)
        return torch.sigmoid(logits).cpu().numpy()


def main():
    seed_everything(SEED)
    torch.set_num_threads(max(1, CPU_THREADS))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== START {EXP_ID} | Member {MEMBER} ===")
    print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")

    start_total = time.time()

    for path in [TRAIN_PATH, TEST_PATH, FOLDS_PATH, SAMPLE_SUB_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    required_train = {ID_COL, TARGET_COL, *CAT_COLS, *BOOL_COLS}
    missing = required_train.difference(train_df.columns)
    if missing:
        raise ValueError(f"Missing required train columns: {sorted(missing)}")

    missing_test = (set(CAT_COLS + BOOL_COLS)).difference(test_df.columns)
    if missing_test:
        raise ValueError(f"Missing required test columns: {sorted(missing_test)}")

    if ID_COL not in folds_df.columns or "fold" not in folds_df.columns:
        raise ValueError("folds.csv must contain 'id' and 'fold' columns.")
    if folds_df[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs found in folds.csv.")

    # Join by ID rather than relying on row order.
    train_df = train_df.merge(
        folds_df[[ID_COL, "fold"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    if train_df["fold"].isna().any():
        raise ValueError("Some training IDs do not have a fold assignment.")

    train_df["fold"] = train_df["fold"].astype(int)
    found_folds = sorted(train_df["fold"].unique().tolist())
    if found_folds != list(range(N_SPLITS)):
        raise ValueError(
            f"Expected fold IDs 0..{N_SPLITS - 1}, found {found_folds}"
        )

    excluded = set(CAT_COLS + [ID_COL, TARGET_COL, "fold"])
    cont_cols = [col for col in train_df.columns if col not in excluded]

    # Convert known boolean columns to numeric; all remaining continuous
    # columns are also safely coerced inside fold-local preprocessing.
    for col in BOOL_COLS:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Categorical columns ({len(CAT_COLS)}): {CAT_COLS}")
    print(f"Continuous columns ({len(cont_cols)}): {len(cont_cols)}")
    print(f"Fraud rate: {train_df[TARGET_COL].mean():.5%}")
    print(f"Fold sizes: {train_df['fold'].value_counts().sort_index().to_dict()}")

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_aucs = []

    for fold in range(N_SPLITS):
        fold_start = time.time()
        print(f"\n========== FOLD {fold + 1}/{N_SPLITS} ==========")

        train_mask = train_df["fold"] != fold
        valid_mask = train_df["fold"] == fold

        train_part = train_df.loc[train_mask].reset_index(drop=True)
        valid_part = train_df.loc[valid_mask].reset_index(drop=True)
        test_part = test_df.reset_index(drop=True)

        x_cat_train, encoded_others, cardinalities = encode_categories(
            train_part, [valid_part, test_part], CAT_COLS
        )
        x_cat_valid, x_cat_test = encoded_others

        x_cont_train, scaled_others = prepare_continuous(
            train_part, [valid_part, test_part], cont_cols
        )
        x_cont_valid, x_cont_test = scaled_others

        y_train = train_part[TARGET_COL].to_numpy(dtype=np.float32)
        y_valid = valid_part[TARGET_COL].to_numpy(dtype=np.int64)

        x_cat_train_t = torch.from_numpy(x_cat_train)
        x_cont_train_t = torch.from_numpy(x_cont_train)
        y_train_t = torch.from_numpy(y_train)

        x_cat_valid_t = torch.from_numpy(x_cat_valid)
        x_cont_valid_t = torch.from_numpy(x_cont_valid)
        x_cat_test_t = torch.from_numpy(x_cat_test)
        x_cont_test_t = torch.from_numpy(x_cont_test)

        train_loader = DataLoader(
            TensorDataset(x_cat_train_t, x_cont_train_t, y_train_t),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,  # avoids BatchNorm receiving a one-row final batch
            num_workers=0,
            pin_memory=False,
        )

        model = ResidualTabularMLP(
            cardinalities=cardinalities,
            num_continuous=len(cont_cols),
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )

        best_auc = -1.0
        best_epoch = 0
        best_state = None
        best_valid_predictions = None
        epochs_without_improvement = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            batches = 0

            for batch_cat, batch_cont, batch_y in train_loader:
                batch_cat = batch_cat.to(DEVICE)
                batch_cont = batch_cont.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_cat, batch_cont)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches += 1

            scheduler.step()

            valid_predictions = predict_probabilities(
                model, x_cat_valid_t, x_cont_valid_t
            )
            epoch_auc = roc_auc_score(y_valid, valid_predictions)
            mean_loss = running_loss / max(1, batches)

            print(
                f"Fold {fold + 1} | Epoch {epoch:02d}/{EPOCHS} "
                f"| Loss: {mean_loss:.5f} | Val AUC: {epoch_auc:.6f}"
            )

            if epoch_auc > best_auc + 1e-6:
                best_auc = epoch_auc
                best_epoch = epoch
                best_valid_predictions = valid_predictions.copy()
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best epoch was {best_epoch}."
                )
                break

        if best_state is None or best_valid_predictions is None:
            raise RuntimeError(f"No best checkpoint was saved for fold {fold}.")

        # Restore the exact checkpoint used to produce the best validation AUC.
        model.load_state_dict(best_state)
        model.eval()

        oof_preds[valid_mask.to_numpy()] = best_valid_predictions
        test_fold_predictions = predict_probabilities(
            model, x_cat_test_t, x_cont_test_t
        )
        test_preds += test_fold_predictions / N_SPLITS

        fold_aucs.append(best_auc)
        elapsed = time.time() - fold_start
        print(
            f"--- Fold {fold + 1}/{N_SPLITS} COMPLETE "
            f"| Best epoch: {best_epoch} | Best AUC: {best_auc:.6f} "
            f"| Time: {elapsed / 60:.1f} min ---"
        )

        del model, optimizer, scheduler, train_loader

    overall_auc = roc_auc_score(train_df[TARGET_COL], oof_preds)
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))

    print("\n" + "=" * 65)
    print(f"FINISHED {EXP_ID}")
    print(f"Fold AUCs: {[round(value, 6) for value in fold_aucs]}")
    print(f"Mean Fold AUC: {mean_auc:.6f} +/- {std_auc:.6f}")
    print(f"Overall OOF AUC: {overall_auc:.6f}")
    print(f"Total time: {(time.time() - start_total) / 60:.1f} min")
    print("=" * 65)

    oof_path = OUTPUT_DIR / f"{EXP_ID}_oof.npy"
    test_path = OUTPUT_DIR / f"{EXP_ID}_test.npy"
    submission_path = SUB_DIR / f"{EXP_ID}.csv"

    np.save(oof_path, oof_preds)
    np.save(test_path, test_preds)

    if len(sample_sub) != len(test_preds):
        raise ValueError(
            f"sample_submission has {len(sample_sub)} rows but test has "
            f"{len(test_preds)} rows."
        )
    if TARGET_COL not in sample_sub.columns:
        raise ValueError(f"'{TARGET_COL}' column missing from sample_submission.")

    sample_sub[TARGET_COL] = test_preds
    sample_sub.to_csv(submission_path, index=False)

    print(f"Saved OOF: {oof_path}")
    print(f"Saved Test: {test_path}")
    print(f"Saved submission: {submission_path}")

    return overall_auc, mean_auc, std_auc


if __name__ == "__main__":
    main()
