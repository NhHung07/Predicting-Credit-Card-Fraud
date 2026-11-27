#!/usr/bin/env python3
"""
Thí nghiệm: Balanced 6-Model Pure ML + DL Probit Ensemble (ens_c_v3_balanced)
Thành viên: C (Phạm Hồng Nam) & Nhóm
Họ mô hình: 100% Pure ML & Deep Learning (0% LLM).
Kết hợp cân bằng trong không gian Gaussian Probit-Rank giữa:
  - lgbm_a_v3 (30%): GLM-Boosted LightGBM [ML]
  - xgb_b_v2  (25%): GLM-Boosted XGBoost GPU [ML]
  - lgbm_a_v4 (20%): Standalone Digit-TE LightGBM [ML]
  - cat_c_v2  (10%): GLM-Boosted CatBoost [ML]
  - mlp_c_v2  (10%): PyTorch Wide&Deep Periodic TabResNet [DL]
  - xgb_b_v3  ( 5%): Standalone Digit-TE XGBoost GPU [ML]
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_C = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"

WEIGHTS = {
    "lgbm_a_v3": (ROOT / "A/outputs/lgbm_a_v3_oof.npy", ROOT / "A/outputs/lgbm_a_v3_test.npy", 0.30),
    "xgb_b_v2": (ROOT / "B/outputs/xgb_b_v2_oof.npy", ROOT / "B/outputs/xgb_b_v2_test.npy", 0.25),
    "lgbm_a_v4": (ROOT / "A/outputs/lgbm_a_v4_oof.npy", ROOT / "A/outputs/lgbm_a_v4_test.npy", 0.20),
    "cat_c_v2": (ROOT / "C/outputs/cat_c_v2_oof.npy", ROOT / "C/outputs/cat_c_v2_test.npy", 0.10),
    "mlp_c_v2": (ROOT / "C/outputs/mlp_c_v2_oof.npy", ROOT / "C/outputs/mlp_c_v2_test.npy", 0.10),
    "xgb_b_v3": (ROOT / "B/outputs/xgb_b_v3_oof.npy", ROOT / "B/outputs/xgb_b_v3_test.npy", 0.05),
}


def to_probit_within_fold(arr: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    for f in np.unique(folds):
        m = folds == f
        out[m] = norm.ppf(np.clip(rankdata(arr[m]) / (m.sum() + 1.0), 1e-6, 1.0 - 1e-6))
    return out


def to_probit_global(arr: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(rankdata(arr) / (len(arr) + 1.0), 1e-6, 1.0 - 1e-6))


def main():
    y_true = pd.read_csv(DATA_DIR / "train.csv", usecols=["is_fraud"])["is_fraud"].values
    folds = pd.read_csv(DATA_DIR / "folds.csv")["fold"].values
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    oof_blend_z = np.zeros(len(y_true), dtype=np.float64)
    test_blend_z = np.zeros(len(sub), dtype=np.float64)

    for name, (oof_p, test_p, w) in WEIGHTS.items():
        oof_blend_z += w * to_probit_within_fold(np.load(oof_p), folds)
        test_blend_z += w * to_probit_global(np.load(test_p))

    oof_prob = norm.cdf(oof_blend_z)
    test_prob = norm.cdf(test_blend_z)

    f_aucs = [roc_auc_score(y_true[folds == f], oof_prob[folds == f]) for f in range(5)]
    print(f"ens_c_v3_balanced | Overall OOF AUC: {roc_auc_score(y_true, oof_prob):.5f} | Mean Fold: {np.mean(f_aucs):.5f} +/- {np.std(f_aucs):.5f}")

    np.save(OUTPUT_C / "ens_c_v3_balanced_oof.npy", oof_prob)
    np.save(OUTPUT_C / "ens_c_v3_balanced_test.npy", test_prob)
    sub["is_fraud"] = test_prob
    sub.to_csv(SUB_DIR / "ens_c_v3_balanced.csv", index=False)
    print("Saved ens_c_v3_balanced OOF, Test, and Submission CSV!")


if __name__ == "__main__":
    main()
