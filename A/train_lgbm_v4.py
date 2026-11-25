#!/usr/bin/env python3
"""
Thí nghiệm: LightGBM v4 - Standalone Shallow LightGBM (num_leaves=6, không dùng GLM init_score)
Thành viên: A (Trần Nhật Hưng)
Mã thí nghiệm: lgbm_a_v4
Họ mô hình: Classical Machine Learning (cấu trúc chia nhánh độc lập để đa dạng hóa Ensemble với v3)
"""

import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from shared.fe_digit_te import compute_fold_te_features, prepare_base_and_keys

EXP_ID = "lgbm_a_v4"
MEMBER = "A"
SEED = 100
N_SPLITS = 5

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "A" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"=== [START] Huấn luyện {EXP_ID} (Standalone Digit-TE LightGBM - Member {MEMBER}) ===")
    t0 = time.time()

    bundle = prepare_base_and_keys(DATA_DIR)
    n_train = bundle["n_train"]
    y_all = bundle["y_all"]
    base_df = bundle["base_df"]
    n_test = len(base_df) - n_train

    oof_preds = np.zeros(n_train, dtype=np.float64)
    test_preds = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.025,
        "num_leaves": 6,
        "min_child_samples": 50,
        "colsample_bytree": 0.4,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_lambda": 2.0,
        "n_estimators": 4000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold in range(N_SPLITS):
        f_start = time.time()
        fe = compute_fold_te_features(bundle, fold, m=15.0, inner_seed=123)
        tr_idx, va_idx, te_idx = fe["tr_idx"], fe["va_idx"], fe["te_idx"]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        te_tr_df = pd.DataFrame(np.hstack([fe["X_te_tr"], fe["F_tr"]]), columns=fe["te_names"] + fe["freq_names"])
        te_va_df = pd.DataFrame(np.hstack([fe["X_te_va"], fe["F_va"]]), columns=fe["te_names"] + fe["freq_names"])
        te_te_df = pd.DataFrame(np.hstack([fe["X_te_te"], fe["F_te"]]), columns=fe["te_names"] + fe["freq_names"])

        X_tr = pd.concat([base_df.iloc[tr_idx].reset_index(drop=True), te_tr_df], axis=1)
        X_va = pd.concat([base_df.iloc[va_idx].reset_index(drop=True), te_va_df], axis=1)
        X_te = pd.concat([base_df.iloc[te_idx].reset_index(drop=True), te_te_df], axis=1)

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        val_prob = model.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = val_prob
        test_preds += model.predict_proba(X_te)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_va, val_prob)
        fold_aucs.append(fold_auc)
        print(
            f"--- Fold {fold+1}/{N_SPLITS} | AUC: {fold_auc:.5f} | "
            f"Best Iter: {model.best_iteration_} | Time: {time.time() - f_start:.1f}s ---"
        )

        del fe, te_tr_df, te_va_df, te_te_df, X_tr, X_va, X_te, model
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
