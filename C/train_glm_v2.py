#!/usr/bin/env python3
"""
Thí nghiệm: Spline/Quantile Digit-TE Generalized Linear Model (GLM - Logistic Regression L2)
Thành viên: C (Phạm Hồng Nam)
Mã thí nghiệm: glm_c_v1
Họ mô hình: Classical Machine Learning (Linear Model trên không gian Logit Target Encoding)
Lưu thêm `glm_logits_cache.npz` (gồm inner-fold out-of-fold train logits và test logits từng fold)
để làm `init_score` / `base_margin` chống rò rỉ dữ liệu cho các mô hình GLM-Boosted LightGBM/XGBoost/CatBoost.
"""

import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from shared.fe_digit_te import compute_fold_te_features, prepare_base_and_keys

EXP_ID = "glm_c_v1"
MEMBER = "C"
SEED = 42
N_SPLITS = 5

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"=== [START] Huấn luyện {EXP_ID} (Digit-TE GLM - Member {MEMBER}) ===")
    t0 = time.time()

    bundle = prepare_base_and_keys(DATA_DIR)
    n_train = bundle["n_train"]
    y_all = bundle["y_all"]
    raw_num_all = bundle["raw_num_all"]
    n_test = len(bundle["base_df"]) - n_train

    oof_preds = np.zeros(n_train, dtype=np.float64)
    oof_logits = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float64)
    test_logits_folds = np.zeros((N_SPLITS, n_test), dtype=np.float32)
    tr_logits_folds = {}
    fold_aucs = []

    for fold in range(N_SPLITS):
        f_start = time.time()
        fe = compute_fold_te_features(bundle, fold, m=15.0, inner_seed=123)
        tr_idx, va_idx, te_idx = fe["tr_idx"], fe["va_idx"], fe["te_idx"]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
        num_tr_qt = qt.fit_transform(raw_num_all[tr_idx]).astype(np.float32)
        num_va_qt = qt.transform(raw_num_all[va_idx]).astype(np.float32)
        num_te_qt = qt.transform(raw_num_all[te_idx]).astype(np.float32)

        X_tr = np.hstack([fe["X_te_tr"], fe["F_tr"], num_tr_qt])
        X_va = np.hstack([fe["X_te_va"], fe["F_va"], num_va_qt])
        X_te = np.hstack([fe["X_te_te"], fe["F_te"], num_te_qt])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)
        X_te_s = scaler.transform(X_te).astype(np.float32)

        # Inner 5-fold CV trên outer train fold để tạo init_score không rò rỉ
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold)
        glm_tr_logit = np.zeros(len(tr_idx), dtype=np.float32)
        for g_tr, g_va in skf_inner.split(X_tr_s, y_tr):
            gm = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
            gm.fit(X_tr_s[g_tr], y_tr[g_tr])
            glm_tr_logit[g_va] = logit(np.clip(gm.predict_proba(X_tr_s[g_va])[:, 1], 1e-6, 1.0 - 1e-6))

        # Fit toàn outer train fold
        glm_full = LogisticRegression(C=0.01, max_iter=500, solver="lbfgs")
        glm_full.fit(X_tr_s, y_tr)

        va_logit = logit(np.clip(glm_full.predict_proba(X_va_s)[:, 1], 1e-6, 1.0 - 1e-6))
        te_logit = logit(np.clip(glm_full.predict_proba(X_te_s)[:, 1], 1e-6, 1.0 - 1e-6))

        va_prob = expit(va_logit)
        oof_preds[va_idx] = va_prob
        oof_logits[va_idx] = va_logit
        test_logits_folds[fold] = te_logit
        test_preds += expit(te_logit) / N_SPLITS
        tr_logits_folds[f"tr_logit_f{fold}"] = glm_tr_logit

        fold_auc = roc_auc_score(y_va, va_prob)
        fold_aucs.append(fold_auc)
        print(f"--- Fold {fold+1}/{N_SPLITS} | GLM AUC: {fold_auc:.5f} | Time: {time.time() - f_start:.1f}s ---")

        del fe, X_tr, X_va, X_te, X_tr_s, X_va_s, X_te_s
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
    np.savez(
        OUTPUT_DIR / "glm_logits_cache.npz",
        oof_logit=oof_logits,
        test_logits_folds=test_logits_folds,
        **tr_logits_folds,
    )

    sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub_df["is_fraud"] = test_preds
    sub_df.to_csv(SUB_DIR / f"{EXP_ID}.csv", index=False)
    print(f"Đã lưu OOF, Test, Logit Cache và Submission: {SUB_DIR / f'{EXP_ID}.csv'}")


if __name__ == "__main__":
    main()
