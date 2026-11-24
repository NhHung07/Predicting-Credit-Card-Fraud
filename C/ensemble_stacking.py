#!/usr/bin/env python3
"""
Thí nghiệm: Ensemble & Stacking Pipeline
Thành viên: C (Phạm Hồng Nam) & Team
Mã thí nghiệm: ens_c_v1
Kết hợp dự đoán OOF và Test của:
  1. LightGBM v2 (A: lgbm_a_v2)
  2. XGBoost v1 (B: xgb_b_v1)
  3. CatBoost v1 (C: cat_c_v1)
Phương pháp so sánh:
  - Simple Average
  - Weighted Blend (Tối ưu SLSQP trên OOF theo ROC-AUC)
  - Rank Averaging (Trung bình phân vị)
  - Stacking Logistic Regression
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EXP_ID = "ens_c_v1"
MEMBER = "C"
SEED = 42

OUTPUT_A = ROOT / "A" / "outputs"
OUTPUT_B = ROOT / "B" / "outputs"
OUTPUT_C = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
DATA_DIR = ROOT / "data"

OUTPUT_C.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== [START] Bắt đầu quy trình Ensemble & Stacking ===")
    start_time = time.time()

    # 1. Đọc nhãn thật và folds
    train_df = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", "is_fraud"])
    sub_sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    folds_df = pd.read_csv(DATA_DIR / "folds.csv")
    y_true = train_df["is_fraud"].values
    folds = folds_df["fold"].values

    # 2. Đọc các OOF và Test predictions
    print("\n==> Tải kết quả OOF và Test Predictions từ các mô hình thành phần...")
    models = {
        "LightGBM v2": {
            "oof": np.load(OUTPUT_A / "lgbm_a_v2_oof.npy"),
            "test": np.load(OUTPUT_A / "lgbm_a_v2_test.npy")
        },
        "XGBoost v1": {
            "oof": np.load(OUTPUT_B / "xgb_b_v1_oof.npy"),
            "test": np.load(OUTPUT_B / "xgb_b_v1_test.npy")
        },
        "CatBoost v1": {
            "oof": np.load(OUTPUT_C / "cat_c_v1_oof.npy"),
            "test": np.load(OUTPUT_C / "cat_c_v1_test.npy")
        }
    }

    # Đánh giá điểm đơn lẻ
    print("\n" + "=" * 50)
    print("Điểm OOF ROC-AUC của từng mô hình đơn lẻ:")
    for name, data in models.items():
        auc = roc_auc_score(y_true, data["oof"])
        print(f"  * {name:<15}: {auc:.5f}")
    print("=" * 50)

    # 3. Phân tích tương quan giữa các mô hình
    oof_df = pd.DataFrame({name: data["oof"] for name, data in models.items()})
    print("\nMa trận tương quan Pearson giữa các OOF Predictions:")
    print(oof_df.corr().round(4))

    # 4. Phương pháp 1: Simple Average
    oof_avg = (models["LightGBM v2"]["oof"] + models["XGBoost v1"]["oof"] + models["CatBoost v1"]["oof"]) / 3.0
    test_avg = (models["LightGBM v2"]["test"] + models["XGBoost v1"]["test"] + models["CatBoost v1"]["test"]) / 3.0
    auc_avg = roc_auc_score(y_true, oof_avg)
    print(f"\n[1] Simple Average OOF AUC: {auc_avg:.5f}")

    # 5. Phương pháp 2: Rank Averaging
    oof_rank = (rankdata(models["LightGBM v2"]["oof"]) + rankdata(models["XGBoost v1"]["oof"]) + rankdata(models["CatBoost v1"]["oof"])) / (3.0 * len(y_true))
    test_rank = (rankdata(models["LightGBM v2"]["test"]) + rankdata(models["XGBoost v1"]["test"]) + rankdata(models["CatBoost v1"]["test"])) / (3.0 * len(sub_sample))
    auc_rank = roc_auc_score(y_true, oof_rank)
    print(f"[2] Rank Average OOF AUC:   {auc_rank:.5f}")

    # 6. Phương pháp 3: Weighted Blending (Tối ưu SLSQP trên OOF)
    oof_matrix = np.column_stack([data["oof"] for data in models.values()])
    test_matrix = np.column_stack([data["test"] for data in models.values()])

    def opt_objective(weights):
        w = weights / np.sum(weights)
        blend = np.dot(oof_matrix, w)
        return -roc_auc_score(y_true, blend)

    init_weights = [1.0 / len(models)] * len(models)
    bounds = [(0, 1)] * len(models)
    res = minimize(opt_objective, init_weights, method="SLSQP", bounds=bounds)
    best_weights = res.x / np.sum(res.x)

    oof_blend = np.dot(oof_matrix, best_weights)
    test_blend = np.dot(test_matrix, best_weights)
    auc_blend = roc_auc_score(y_true, oof_blend)

    print(f"[3] Optimized Weighted Blend OOF AUC: {auc_blend:.5f}")
    print("    Trọng số tối ưu:")
    for name, w in zip(models.keys(), best_weights):
        print(f"      - {name}: {w:.4f}")

    # 7. Phương pháp 4: Stacking Meta-Learner (Logistic Regression với Out-of-fold)
    n_splits = len(np.unique(folds))
    oof_stack = np.zeros(len(train_df), dtype=np.float64)
    test_stack = np.zeros(len(sub_sample), dtype=np.float64)

    for fold in range(n_splits):
        tr_idx = folds != fold
        va_idx = folds == fold
        meta_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        meta_clf.fit(oof_matrix[tr_idx], y_true[tr_idx])
        oof_stack[va_idx] = meta_clf.predict_proba(oof_matrix[va_idx])[:, 1]
        test_stack += meta_clf.predict_proba(test_matrix)[:, 1] / n_splits

    auc_stack = roc_auc_score(y_true, oof_stack)
    print(f"[4] Stacking (Logistic Regression) OOF AUC: {auc_stack:.5f}")

    # 8. Chọn phương pháp tối ưu nhất
    comparison = {
        "Simple Average": (auc_avg, oof_avg, test_avg),
        "Rank Average": (auc_rank, oof_rank, test_rank),
        "Weighted Blend": (auc_blend, oof_blend, test_blend),
        "Stacking Logistic": (auc_stack, oof_stack, test_stack)
    }

    best_method = max(comparison.keys(), key=lambda k: comparison[k][0])
    best_auc, final_oof, final_test = comparison[best_method]

    print("\n" + "=" * 60)
    print(f"PHƯƠNG PHÁP ENSEMBLE TỐI ƯU NHẤT: {best_method}")
    print(f">> BEST ENSEMBLE OOF AUC: {best_auc:.5f} <<")
    print("=" * 60)

    # 9. Lưu artifacts
    np.save(OUTPUT_C / f"{EXP_ID}_oof.npy", final_oof)
    np.save(OUTPUT_C / f"{EXP_ID}_test.npy", final_test)
    print(f"Đã lưu OOF: {OUTPUT_C / f'{EXP_ID}_oof.npy'}")
    print(f"Đã lưu Test: {OUTPUT_C / f'{EXP_ID}_test.npy'}")

    # Lưu submission chính (ens_c_v1.csv)
    sub_sample["is_fraud"] = final_test
    sub_file = SUB_DIR / f"{EXP_ID}.csv"
    sub_sample.to_csv(sub_file, index=False)
    print(f"Đã lưu Submission chính: {sub_file}")

    # Lưu thêm bản Rank Average (ens_c_v2_rank.csv) làm phương án dự phòng/đa dạng hoá
    sub_rank = sub_sample.copy()
    sub_rank["is_fraud"] = test_rank
    sub_rank_file = SUB_DIR / "ens_c_v2_rank.csv"
    sub_rank.to_csv(sub_rank_file, index=False)
    print(f"Đã lưu Submission Rank Average: {sub_rank_file}")

    print(f"\nThời gian chạy Ensemble: {(time.time() - start_time):.2f}s")
    return best_method, best_auc

if __name__ == "__main__":
    main()
