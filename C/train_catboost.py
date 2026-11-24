#!/usr/bin/env python3
"""
Thí nghiệm: CatBoost Model đã tinh chỉnh (Tuned CatBoost 5-Fold Stratified)
Thành viên: C (Phạm Hồng Nam)
Mã thí nghiệm: cat_c_v1
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "cat_c_v1"
MEMBER = "C"
SEED = 42
N_SPLITS = 5

TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
FOLDS_PATH = ROOT / "data" / "folds.csv"
SUB_SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"

OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"=== [START] Huấn luyện {EXP_ID} (Member {MEMBER}) ===")
    start_total = time.time()

    print("==> Đọc dữ liệu...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)
    sub_df = pd.read_csv(SUB_SAMPLE_PATH)

    train_df["fold"] = folds_df["fold"]

    cat_cols = [
        "merchant_category",
        "card_type",
        "auth_method",
        "channel",
        "device_type"
    ]
    bool_cols = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt"
    ]

    for c in bool_cols:
        train_df[c] = train_df[c].astype(int)
        test_df[c] = test_df[c].astype(int)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = oe.fit_transform(train_df[cat_cols])
    test_df[cat_cols] = oe.transform(test_df[cat_cols])

    features = [c for c in train_df.columns if c not in ["id", "is_fraud", "fold"]]
    print(f"Số lượng đặc trưng sử dụng: {len(features)}")
    print(f"Danh sách đặc trưng: {features}")

    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.06,
        "depth": 4,
        "l2_leaf_reg": 3.0,
        "iterations": 1500,
        "thread_count": -1,
        "random_seed": SEED,
        "verbose": False,
        "early_stopping_rounds": 50
    }

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64)
    fold_aucs = []

    print("\n==> Bắt đầu huấn luyện 5 Folds Cross Validation với CatBoost...")
    for fold in range(N_SPLITS):
        fold_start = time.time()
        tr_mask = train_df["fold"] != fold
        va_mask = train_df["fold"] == fold

        X_train, y_train = train_df.loc[tr_mask, features], train_df.loc[tr_mask, "is_fraud"]
        X_val, y_val = train_df.loc[va_mask, features], train_df.loc[va_mask, "is_fraud"]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[va_mask] = val_pred

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)

        test_preds += model.predict_proba(test_df[features])[:, 1] / N_SPLITS

        best_iter = model.get_best_iteration()
        fold_elapsed = time.time() - fold_start
        print(f"--- Fold {fold+1}/{N_SPLITS} | AUC: {fold_auc:.5f} | Best Iter: {best_iter} | Time: {fold_elapsed:.1f}s ---")

    overall_auc = roc_auc_score(train_df["is_fraud"], oof_preds)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    total_elapsed = (time.time() - start_total) / 60.0

    print("=" * 60)
    print(f"HOÀN THÀNH HUẤN LUYỆN {EXP_ID}")
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

    print(f"Tóm tắt phân phối xác suất dự đoán test:")
    print(f"  Min:  {test_preds.min():.6f}")
    print(f"  Mean: {test_preds.mean():.6f}")
    print(f"  Max:  {test_preds.max():.6f}")

    return overall_auc, mean_auc, std_auc

if __name__ == "__main__":
    main()
