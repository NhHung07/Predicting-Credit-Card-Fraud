#!/usr/bin/env python3
"""
Thí nghiệm: XGBoost v3 - Standalone Shallow GPU Trees (max_depth=3, không dùng GLM base_margin)
Thành viên: B (Cao Anh Minh)
Mã thí nghiệm: xgb_b_v3
Họ mô hình: Classical Machine Learning (tối ưu bổ trợ đa dạng với xgb_b_v2)
"""

import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from shared.fe_digit_te import CAT_COLS, compute_fold_te_features, prepare_base_and_keys

EXP_ID = "xgb_b_v3"
MEMBER = "B"
SEED = 100
N_SPLITS = 5

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "B" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"=== [START] Huấn luyện {EXP_ID} (Standalone Digit-TE XGBoost GPU - Member {MEMBER}) ===")
    t0 = time.time()

    bundle = prepare_base_and_keys(DATA_DIR)
    n_train = bundle["n_train"]
    y_all = bundle["y_all"]
    base_df = bundle["base_df"].copy()
    for c in CAT_COLS:
        base_df[c] = base_df[c].cat.codes.astype(np.int32)
    n_test = len(base_df) - n_train

    oof_preds = np.zeros(n_train, dtype=np.float64)
    test_preds = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "reg_lambda": 2.0,
        "n_estimators": 4000,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "early_stopping_rounds": 100,
    }

    for fold in range(N_SPLITS):
        f_start = time.time()
        fe = compute_fold_te_features(bundle, fold, m=15.0, inner_seed=123)
        tr_idx, va_idx, te_idx = fe["tr_idx"], fe["va_idx"], fe["te_idx"]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        X_tr = np.hstack([base_df.iloc[tr_idx].values.astype(np.float32), fe["X_te_tr"], fe["F_tr"]])
        X_va = np.hstack([base_df.iloc[va_idx].values.astype(np.float32), fe["X_te_va"], fe["F_va"]])
        X_te = np.hstack([base_df.iloc[te_idx].values.astype(np.float32), fe["X_te_te"], fe["F_te"]])

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        val_prob = model.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = val_prob
        test_preds += model.predict_proba(X_te)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_va, val_prob)
        fold_aucs.append(fold_auc)
        print(
            f"--- Fold {fold+1}/{N_SPLITS} | AUC: {fold_auc:.5f} | "
            f"Best Iter: {model.best_iteration} | Time: {time.time() - f_start:.1f}s ---"
        )

        del fe, X_tr, X_va, X_te, model
        gc.collect()

    overall_auc = roc_auc_score(y_all, oof_preds)
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print("=" * 60)
    print(f"HOÀN THÀNH {EXP_ID} | Tổng thời gian: {(time.time() - t0) / 60.0:.2f} phút")
    print(f"AUC từng fold: {[round(x, 5) for x in fold_aucs]}")
    print(f"Mean Fold AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f">> OVERALL OOF AUC: {overall_auc:.5f} <<")
    print("=" * 60)

    np.save(OUTPUT_DIR / f"{EXP_ID}_oof.npy", oof_preds)
    np.save(OUTPUT_DIR / f"{EXP_ID}_test.npy", test_preds)

    sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub_df["is_fraud"] = test_preds
    sub_df.to_csv(SUB_DIR / f"{EXP_ID}.csv", index=False)
    print(f"Đã lưu OOF, Test và Submission: {SUB_DIR / f'{EXP_ID}.csv'}")


if __name__ == "__main__":
    main()
