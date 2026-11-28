
#!/usr/bin/env python3
"""
CatBoost v3 - Nonlinear Digit-TE Ensemble
Member: Phạm Hồng Nam
Experiment: cat_c_v3

- No GLM baseline / no GLM logits.
- Leak-aware Digit-TE and Frequency features.
- Train depth 5 and depth 6 on the same 5 folds.
- Save individual OOF/test predictions and their average blend.
"""

import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from shared.fe_digit_te import (
    CAT_COLS,
    compute_fold_te_features,
    prepare_base_and_keys,
)

EXP_ID = "cat_c_v3"
SEED = 42
N_SPLITS = 5

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

# Hai cấu hình có độ phức tạp khác nhau để kiểm tra khả năng tổng quát hóa.
MODEL_CONFIGS = {
    "d5": {
        "depth": 5,
        "learning_rate": 0.03,
        "iterations": 4500,
        "l2_leaf_reg": 6.0,
        "random_strength": 0.5,
        "bagging_temperature": 0.5,
    },
    "d6": {
        "depth": 6,
        "learning_rate": 0.03,
        "iterations": 4500,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.7,
        "bagging_temperature": 0.7,
    },
}


def build_model(config, seed):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "CPU",
        "boosting_type": "Plain",
        "border_count": 128,
        "thread_count": -1,
        "random_seed": seed,
        "early_stopping_rounds": 180,
        "verbose": 250,
        "allow_writing_files": False,
    }
    params.update(config)
    return CatBoostClassifier(**params)


def save_submission(predictions, filename):
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if len(sample) != len(predictions):
        raise ValueError(
            f"Submission length mismatch: {len(sample)} != {len(predictions)}"
        )

    sample["is_fraud"] = np.clip(predictions, 0.0, 1.0)
    path = SUB_DIR / filename
    sample.to_csv(path, index=False)
    print(f"Saved submission: {path}")


def main():
    start_time = time.time()

    print("=" * 70)
    print(f"START {EXP_ID}")
    print("No GLM baseline | Digit-TE + Frequency | Depth 5 and 6")
    print("=" * 70)

    bundle = prepare_base_and_keys(DATA_DIR)

    n_train = bundle["n_train"]
    y_all = np.asarray(bundle["y_all"], dtype=np.int8)

    base_df = bundle["base_df"].copy()

    # Giữ đúng cách mã hóa categorical như pipeline v2.
    for col in CAT_COLS:
        base_df[col] = base_df[col].cat.codes.astype(np.int32)

    n_test = len(base_df) - n_train

    if len(y_all) != n_train:
        raise ValueError("Target length does not match n_train.")

    oof = {
        name: np.full(n_train, np.nan, dtype=np.float64)
        for name in MODEL_CONFIGS
    }
    test_pred = {
        name: np.zeros(n_test, dtype=np.float64)
        for name in MODEL_CONFIGS
    }

    fold_scores = {name: [] for name in MODEL_CONFIGS}

    for fold in range(N_SPLITS):
        fold_start = time.time()

        print("\n" + "=" * 70)
        print(f"FOLD {fold + 1}/{N_SPLITS}")
        print("=" * 70)

        # Tạo TE/Frequency features theo đúng outer fold.
        fe = compute_fold_te_features(
            bundle,
            fold,
            m=15.0,
            inner_seed=123,
        )

        tr_idx = fe["tr_idx"]
        va_idx = fe["va_idx"]
        te_idx = fe["te_idx"]

        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        X_tr = np.hstack([
            base_df.iloc[tr_idx].to_numpy(dtype=np.float32),
            fe["X_te_tr"].astype(np.float32),
            fe["F_tr"].astype(np.float32),
        ])

        X_va = np.hstack([
            base_df.iloc[va_idx].to_numpy(dtype=np.float32),
            fe["X_te_va"].astype(np.float32),
            fe["F_va"].astype(np.float32),
        ])

        X_te = np.hstack([
            base_df.iloc[te_idx].to_numpy(dtype=np.float32),
            fe["X_te_te"].astype(np.float32),
            fe["F_te"].astype(np.float32),
        ])

        if X_tr.shape[1] != X_va.shape[1] or X_tr.shape[1] != X_te.shape[1]:
            raise ValueError("Feature dimension mismatch between train/val/test.")

        print(
            f"Train: {X_tr.shape} | "
            f"Valid: {X_va.shape} | "
            f"Test: {X_te.shape}"
        )

        pool_tr = Pool(X_tr, label=y_tr)
        pool_va = Pool(X_va, label=y_va)
        pool_te = Pool(X_te)

        for model_name, config in MODEL_CONFIGS.items():
            print(f"\n--- Training {model_name.upper()} ---")

            model = build_model(
                config,
                seed=SEED + fold * 100 + config["depth"],
            )

            model.fit(
                pool_tr,
                eval_set=pool_va,
                use_best_model=True,
                verbose=250,
            )

            val_prob = model.predict_proba(pool_va)[:, 1]
            test_prob = model.predict_proba(pool_te)[:, 1]

            oof[model_name][va_idx] = val_prob
            test_pred[model_name] += test_prob / N_SPLITS

            fold_auc = roc_auc_score(y_va, val_prob)
            fold_scores[model_name].append(fold_auc)

            print(
                f"{model_name.upper()} | "
                f"Fold AUC: {fold_auc:.6f} | "
                f"Best iteration: {model.get_best_iteration()} | "
                f"Trees: {model.tree_count_}"
            )

            del model, val_prob, test_prob
            gc.collect()

        del fe, X_tr, X_va, X_te
        del pool_tr, pool_va, pool_te
        gc.collect()

        print(f"Fold time: {(time.time() - fold_start) / 60:.2f} minutes")

    print("\n" + "=" * 70)
    print("FINAL OOF EVALUATION")
    print("=" * 70)

    for model_name in MODEL_CONFIGS:
        if np.isnan(oof[model_name]).any():
            raise ValueError(f"OOF predictions incomplete for {model_name}.")

        overall_auc = roc_auc_score(y_all, oof[model_name])
        mean_auc = np.mean(fold_scores[model_name])
        std_auc = np.std(fold_scores[model_name])

        print(
            f"{model_name.upper()} | "
            f"Overall OOF AUC: {overall_auc:.6f} | "
            f"Mean Fold AUC: {mean_auc:.6f} +/- {std_auc:.6f}"
        )

        np.save(
            OUTPUT_DIR / f"{EXP_ID}_{model_name}_oof.npy",
            oof[model_name],
        )
        np.save(
            OUTPUT_DIR / f"{EXP_ID}_{model_name}_test.npy",
            test_pred[model_name],
        )

        save_submission(
            test_pred[model_name],
            f"{EXP_ID}_{model_name}.csv",
        )

    # Blend hai cấu hình: giữ riêng từng mô hình để so sánh.
    blend_oof = (oof["d5"] + oof["d6"]) / 2.0
    blend_test = (test_pred["d5"] + test_pred["d6"]) / 2.0

    blend_auc = roc_auc_score(y_all, blend_oof)

    print("-" * 70)
    print(f"BLEND D5 + D6 | Overall OOF AUC: {blend_auc:.6f}")
    print("-" * 70)

    np.save(OUTPUT_DIR / f"{EXP_ID}_blend_oof.npy", blend_oof)
    np.save(OUTPUT_DIR / f"{EXP_ID}_blend_test.npy", blend_test)

    save_submission(blend_test, f"{EXP_ID}.csv")

    print("\n" + "=" * 70)
    print(f"FINISHED {EXP_ID}")
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Submission: {SUB_DIR / f'{EXP_ID}.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
