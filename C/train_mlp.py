#!/usr/bin/env python3
"""
Thí nghiệm: Deep Learning - PyTorch Tabular Multi-Layer Perceptron (5-Fold Stratified)
Thành viên: C (Phạm Hồng Nam)
Mã thí nghiệm: mlp_c_v1
Thuộc họ: Deep Learning (DL) thuần túy (Entity Embedding + BatchNorm + ResMLP)
100% Không sử dụng bất kỳ LLM hay Foundation Model nào.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "mlp_c_v1"
MEMBER = "C"
SEED = 42
N_SPLITS = 5
EPOCHS = 6
BATCH_SIZE = 4096
LR = 3e-3

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "data" / "folds.csv"
SUB_SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"

OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabularMLP(nn.Module):
    def __init__(self, emb_dims, num_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(num_cat, dim) for num_cat, dim in emb_dims])
        total_emb_dim = sum(dim for _, dim in emb_dims)
        in_dim = total_emb_dim + num_cont
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_cat, x_cont):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(emb_outs + [x_cont], dim=1)
        return self.mlp(x).squeeze(1)

def main():
    print(f"=== [START] Huấn luyện {EXP_ID} Deep Learning (Member {MEMBER}) ===")
    print(f"Thiết bị tính toán: {device}")
    start_total = time.time()

    print("==> Đọc và tiền xử lý dữ liệu...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)
    sub_df = pd.read_csv(SUB_SAMPLE_PATH)

    train_df["fold"] = folds_df["fold"]

    cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
    bool_cols = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt"
    ]
    num_cols = [c for c in train_df.columns if c not in cat_cols + bool_cols + ["id", "is_fraud", "fold"]]

    for c in bool_cols:
        train_df[c] = train_df[c].astype(float)
        test_df[c] = test_df[c].astype(float)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = oe.fit_transform(train_df[cat_cols]).astype(int)
    test_df[cat_cols] = oe.transform(test_df[cat_cols]).astype(int)

    scaler = StandardScaler()
    cont_cols = num_cols + bool_cols
    train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    cat_cardinalities = [train_df[c].nunique() for c in cat_cols]
    emb_dims = [(num_cat, min(16, (num_cat + 1) // 2 + 2)) for num_cat in cat_cardinalities]

    print(f"Số lượng biến liên tục (Continuous): {len(cont_cols)}")
    print(f"Số lượng biến định tính (Categorical Embedding): {len(cat_cols)}")

    X_cat_test = torch.tensor(test_df[cat_cols].values, dtype=torch.long)
    X_cont_test = torch.tensor(test_df[cont_cols].values, dtype=torch.float32)

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_aucs = []

    print("\n==> Bắt đầu huấn luyện 5 Folds Cross Validation với PyTorch Deep Neural Network...")
    for fold in range(N_SPLITS):
        fold_start = time.time()
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        X_cat_tr = torch.tensor(train_df.loc[tr_mask, cat_cols].values, dtype=torch.long)
        X_cont_tr = torch.tensor(train_df.loc[tr_mask, cont_cols].values, dtype=torch.float32)
        y_tr = torch.tensor(train_df.loc[tr_mask, "is_fraud"].values, dtype=torch.float32)

        X_cat_va = torch.tensor(train_df.loc[va_mask, cat_cols].values, dtype=torch.long)
        X_cont_va = torch.tensor(train_df.loc[va_mask, cont_cols].values, dtype=torch.float32)
        y_va = train_df.loc[va_mask, "is_fraud"].values

        train_loader = DataLoader(
            TensorDataset(X_cat_tr, X_cont_tr, y_tr),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True
        )

        model = TabularMLP(emb_dims, len(cont_cols)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_auc = 0.0
        best_val_preds = None

        for epoch in range(EPOCHS):
            model.train()
            for cat_b, cont_b, y_b in train_loader:
                cat_b = cat_b.to(device, non_blocking=True)
                cont_b = cont_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)

                optimizer.zero_grad()
                out = model(cat_b, cont_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Đánh giá validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_cat_va.to(device), X_cont_va.to(device)).cpu()
                val_pred_epoch = torch.sigmoid(val_logits).numpy()
                epoch_auc = roc_auc_score(y_va, val_pred_epoch)
                if epoch_auc > best_val_auc:
                    best_val_auc = epoch_auc
                    best_val_preds = val_pred_epoch

        oof_preds[va_mask] = best_val_preds
        fold_aucs.append(best_val_auc)

        # Dự đoán test
        model.eval()
        with torch.no_grad():
            test_logits = model(X_cat_test.to(device), X_cont_test.to(device)).cpu()
            test_preds += torch.sigmoid(test_logits).numpy() / N_SPLITS

        fold_elapsed = time.time() - fold_start
        print(f"--- Fold {fold+1}/{N_SPLITS} | AUC: {best_val_auc:.5f} | Time: {fold_elapsed:.1f}s ---")

    overall_auc = roc_auc_score(train_df["is_fraud"], oof_preds)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    total_elapsed = (time.time() - start_total) / 60.0

    print("=" * 60)
    print(f"HOÀN THÀNH HUẤN LUYỆN {EXP_ID} (Deep Learning)")
    print(f"Tổng thời gian: {total_elapsed:.2f} phút")
    print(f"AUC từng fold: {[round(x, 5) for x in fold_aucs]}")
    print(f"Mean Fold AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f">> OVERALL OOF AUC: {overall_auc:.5f} <<")
    print("=" * 60)

    oof_file = OUTPUT_DIR / f"{EXP_ID}_oof.npy"
    pred_file = OUTPUT_DIR / f"{EXP_ID}_test.npy"
    np.save(oof_file, oof_preds)
    np.save(pred_file, test_preds)
    print(f"Đã lưu OOF: {oof_file} (shape={oof_preds.shape})")
    print(f"Đã lưu Test: {pred_file} (shape={test_preds.shape})")

    sub_df["is_fraud"] = test_preds
    sub_file = SUB_DIR / f"{EXP_ID}.csv"
    sub_df.to_csv(sub_file, index=False)
    print(f"Đã lưu Submission: {sub_file}")

    return overall_auc, mean_auc, std_auc

if __name__ == "__main__":
    main()
