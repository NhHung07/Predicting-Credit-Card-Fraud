#!/usr/bin/env python3
"""
Thí nghiệm: Pure ML + Deep Learning Mega-Ensemble & Stacking Pipeline (v2)
Thành viên: C (Phạm Hồng Nam) & Nhóm
Mã thí nghiệm: ens_c_v2
Họ mô hình: 100% Pure ML & Deep Learning (0% LLM).
Kết hợp toàn bộ hệ sinh thái mô hình nội bộ:
  - Raw GBDTs: lgbm_a_v2, xgb_b_v1, cat_c_v1
  - Linear Digit-TE ML: glm_c_v1
  - GLM-Boosted Digit-TE GBDTs: lgbm_a_v3, xgb_b_v2, cat_c_v2
  - Standalone Digit-TE GBDTs: lgbm_a_v4, xgb_b_v3
  - Deep Learning (PyTorch GPU): mlp_c_v2 (Wide & Deep Periodic TabResNet)
Chiến lược tối ưu tốc độ cao & chống overfit (đánh giá bằng Nested 5-Fold CV):
  1. Caruana Greedy Hill-Climbing (Ensemble Selection) trên không gian Logit & Probit-Rank.
  2. Non-Negative Ridge Meta-Learner (`positive=True`) trên không gian Probit-Rank.
  3. Logistic Regression Meta-Stacker (`C=0.05`) trên không gian Probit-Rank & Logit.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "ens_c_v2"
MEMBER = "C"
SEED = 42

DATA_DIR = ROOT / "data"
OUTPUT_A = ROOT / "A" / "outputs"
OUTPUT_B = ROOT / "B" / "outputs"
OUTPUT_C = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_C.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


def to_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1.0 - 1e-6))


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Tính ROC-AUC nhanh hơn sklearn bằng sắp xếp thứ tự."""
    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_true) - n_pos
    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    return float((ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def hill_climb_weights(X: np.ndarray, y: np.ndarray, n_steps: int = 40) -> np.ndarray:
    """Caruana Ensemble Selection tìm trọng số tối ưu ROC-AUC."""
    n_models = X.shape[1]
    aucs = [fast_auc(y, X[:, j]) for j in range(n_models)]
    best_idx = int(np.argmax(aucs))
    counts = np.zeros(n_models, dtype=np.float64)
    counts[best_idx] = 1.0
    curr_sum = X[:, best_idx].copy()

    for step in range(2, n_steps + 1):
        best_step_auc = -1.0
        best_j = 0
        for j in range(n_models):
            cand = (curr_sum + X[:, j]) / step
            a = fast_auc(y, cand)
            if a > best_step_auc:
                best_step_auc = a
                best_j = j
        curr_sum += X[:, best_j]
        counts[best_j] += 1.0

    return counts / counts.sum()


def main():
    print("=== [START] Pipeline Ensemble & Stacking v2 (100% Pure ML + DL) ===")
    t0 = time.time()

    train_df = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", "is_fraud"])
    sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
    folds = pd.read_csv(DATA_DIR / "folds.csv")["fold"].values
    y_true = train_df["is_fraud"].values
    n_splits = len(np.unique(folds))

    model_specs = [
        ("lgbm_a_v2", OUTPUT_A / "lgbm_a_v2_oof.npy", OUTPUT_A / "lgbm_a_v2_test.npy", "Raw LightGBM [ML]"),
        ("xgb_b_v1", OUTPUT_B / "xgb_b_v1_oof.npy", OUTPUT_B / "xgb_b_v1_test.npy", "Raw XGBoost [ML]"),
        ("cat_c_v1", OUTPUT_C / "cat_c_v1_oof.npy", OUTPUT_C / "cat_c_v1_test.npy", "Raw CatBoost [ML]"),
        ("glm_c_v1", OUTPUT_C / "glm_c_v1_oof.npy", OUTPUT_C / "glm_c_v1_test.npy", "Digit-TE GLM [ML]"),
        ("lgbm_a_v3", OUTPUT_A / "lgbm_a_v3_oof.npy", OUTPUT_A / "lgbm_a_v3_test.npy", "GLM-Boosted LGBM [ML]"),
        ("lgbm_a_v4", OUTPUT_A / "lgbm_a_v4_oof.npy", OUTPUT_A / "lgbm_a_v4_test.npy", "Standalone Digit-TE LGBM [ML]"),
        ("xgb_b_v2", OUTPUT_B / "xgb_b_v2_oof.npy", OUTPUT_B / "xgb_b_v2_test.npy", "GLM-Boosted XGB [ML]"),
        ("xgb_b_v3", OUTPUT_B / "xgb_b_v3_oof.npy", OUTPUT_B / "xgb_b_v3_test.npy", "Standalone Digit-TE XGB [ML]"),
        ("cat_c_v2", OUTPUT_C / "cat_c_v2_oof.npy", OUTPUT_C / "cat_c_v2_test.npy", "GLM-Boosted CatBoost [ML]"),
        ("mlp_c_v2", OUTPUT_C / "mlp_c_v2_oof.npy", OUTPUT_C / "mlp_c_v2_test.npy", "PyTorch Wide&Deep Periodic TabResNet [DL]"),
    ]

    names, oofs, tests = [], [], []
    print("\n" + "=" * 78)
    print(f"{'Model ID':<12} | {'Architecture':<42} | {'OOF AUC':<8} | {'Mean Fold':<9}")
    print("-" * 78)
    for name, oof_p, test_p, desc in model_specs:
        oof_arr = np.load(oof_p)
        test_arr = np.load(test_p)
        names.append(name)
        oofs.append(oof_arr)
        tests.append(test_arr)
        oof_auc = roc_auc_score(y_true, oof_arr)
        f_mean = np.mean([roc_auc_score(y_true[folds == f], oof_arr[folds == f]) for f in range(n_splits)])
        print(f"{name:<12} | {desc:<42} | {oof_auc:.5f}  | {f_mean:.5f}")
    print("=" * 78)

    M_oof_logit = np.column_stack([to_logit(o) for o in oofs])
    M_test_logit = np.column_stack([to_logit(t) for t in tests])

    # Within-fold Probit Rank chuẩn hóa (loại bỏ trôi calibration giữa các fold)
    M_oof_probit = np.zeros_like(M_oof_logit)
    for f in range(n_splits):
        m = folds == f
        nf = m.sum()
        for j in range(len(names)):
            r = rankdata(oofs[j][m]) / (nf + 1.0)
            M_oof_probit[m, j] = norm.ppf(np.clip(r, 1e-6, 1.0 - 1e-6))

    nt = len(sub_df)
    M_test_probit = np.column_stack(
        [norm.ppf(np.clip(rankdata(t) / (nt + 1.0), 1e-6, 1.0 - 1e-6)) for t in tests]
    )

    # 1. Nested-CV Hill-Climbing trên Logit
    oof_hc_logit = np.zeros(len(y_true), dtype=np.float64)
    test_hc_logit = np.zeros(nt, dtype=np.float64)
    ws_hc_l = []
    for f in range(n_splits):
        tr, va = folds != f, folds == f
        w = hill_climb_weights(M_oof_logit[tr], y_true[tr], n_steps=40)
        ws_hc_l.append(w)
        oof_hc_logit[va] = expit(M_oof_logit[va] @ w)
        test_hc_logit += expit(M_test_logit @ w) / n_splits

    # 2. Nested-CV Hill-Climbing trên Probit-Rank
    oof_hc_probit = np.zeros(len(y_true), dtype=np.float64)
    test_hc_probit = np.zeros(nt, dtype=np.float64)
    ws_hc_p = []
    for f in range(n_splits):
        tr, va = folds != f, folds == f
        w = hill_climb_weights(M_oof_probit[tr], y_true[tr], n_steps=40)
        ws_hc_p.append(w)
        oof_hc_probit[va] = norm.cdf(M_oof_probit[va] @ w)
        test_hc_probit += norm.cdf(M_test_probit @ w) / n_splits

    # 3. Nested-CV Non-Negative Ridge trên Probit-Rank
    oof_ridge = np.zeros(len(y_true), dtype=np.float64)
    test_ridge = np.zeros(nt, dtype=np.float64)
    ws_ridge = []
    for f in range(n_splits):
        tr, va = folds != f, folds == f
        rg = Ridge(alpha=100.0, positive=True, random_state=SEED)
        rg.fit(M_oof_probit[tr], y_true[tr])
        w = rg.coef_ / rg.coef_.sum()
        ws_ridge.append(w)
        oof_ridge[va] = norm.cdf(M_oof_probit[va] @ w)
        test_ridge += norm.cdf(M_test_probit @ w) / n_splits

    # 4. Nested-CV Logistic Regression Meta-Stacker trên Probit-Rank
    oof_stack = np.zeros(len(y_true), dtype=np.float64)
    test_stack = np.zeros(nt, dtype=np.float64)
    for f in range(n_splits):
        tr, va = folds != f, folds == f
        clf = LogisticRegression(C=0.05, max_iter=1000, random_state=SEED)
        clf.fit(M_oof_probit[tr], y_true[tr])
        oof_stack[va] = clf.predict_proba(M_oof_probit[va])[:, 1]
        test_stack += clf.predict_proba(M_test_probit)[:, 1] / n_splits

    candidates = {
        "Hill-Climb Logit Blend": (oof_hc_logit, test_hc_logit, ws_hc_l),
        "Hill-Climb Probit-Rank": (oof_hc_probit, test_hc_probit, ws_hc_p),
        "Non-Neg Ridge Probit": (oof_ridge, test_ridge, ws_ridge),
        "Meta-Logistic Probit": (oof_stack, test_stack, None),
    }

    print("\n=== Kết quả Nested-CV các chiến lược Ensemble ===")
    results = {}
    for cname, (oof_c, test_c, ws_c) in candidates.items():
        auc_c = roc_auc_score(y_true, oof_c)
        mf_c = float(np.mean([roc_auc_score(y_true[folds == f], oof_c[folds == f]) for f in range(n_splits)]))
        std_c = float(np.std([roc_auc_score(y_true[folds == f], oof_c[folds == f]) for f in range(n_splits)]))
        results[cname] = (auc_c, mf_c, std_c, oof_c, test_c)
        print(f"  * {cname:<24}: OOF AUC = {auc_c:.5f} | Mean Fold = {mf_c:.5f} +/- {std_c:.5f}")
        if ws_c is not None:
            avg_w = {k: round(float(v), 4) for k, v in zip(names, np.mean(ws_c, axis=0)) if v > 0.001}
            print(f"      Trọng số TB: {avg_w}")

    best_name = max(results.keys(), key=lambda k: results[k][0])
    best_oof_auc, best_mf_auc, best_std, final_oof, final_test = results[best_name]

    print("\n" + "=" * 68)
    print(f"PHƯƠNG PHÁP ENSEMBLE TỐI ƯU NHẤT: {best_name}")
    print(f">> OVERALL OOF AUC: {best_oof_auc:.5f} | Mean Fold: {best_mf_auc:.5f} +/- {best_std:.5f} <<")
    print("=" * 68)

    np.save(OUTPUT_C / f"{EXP_ID}_oof.npy", final_oof)
    np.save(OUTPUT_C / f"{EXP_ID}_test.npy", final_test)

    sub_df["is_fraud"] = final_test
    sub_file = SUB_DIR / f"{EXP_ID}.csv"
    sub_df.to_csv(sub_file, index=False)
    print(f"Đã lưu OOF, Test và Submission chính thức: {sub_file}")
    print(f"Thời gian chạy: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
