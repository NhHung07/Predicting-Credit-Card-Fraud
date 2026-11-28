#!/usr/bin/env python3
"""
Pipeline Feature Engineering & Leak-Free Digit Target Encoding dùng chung cho cả nhóm.
- Trích xuất đặc trưng chữ số (Digit Keys) cho 6 biến thực: exact, cents, tenths, hundredths,
  int, div10, div100, div1000, 10x, last_d, last2_d, first1, first2, first3.
- Tạo các biến tương tác miền bài toán (domain ratios, bool bitmask, risk_flags_count, 19 cặp tương tác).
- Tính m-estimate Target Encoding (logit-transformed) hoàn toàn chống rò rỉ dữ liệu:
  + Tập train ngoài (outer train fold) sử dụng inner 5-fold StratifiedKFold.
  + Tập validation ngoài (outer val fold) và tập test sử dụng thống kê từ toàn bộ outer train fold.
- Tính Frequency Encoding (log1p count) chỉ trên outer train fold (không rò rỉ tập val/test).
"""

import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.special import logit

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
FLOAT_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]
INT_COLS = [
    "txn_count_last_24h",
    "card_age_months",
    "customer_age",
    "cvv_retry_count",
    "time_of_day_hour",
    "day_of_week",
    "prior_disputes",
]

PAIR_DEFS = [
    ("merchant_category", "auth_method"),
    ("merchant_category", "channel"),
    ("merchant_category", "device_type"),
    ("merchant_category", "cvv_retry_count"),
    ("merchant_category", "is_foreign_transaction"),
    ("merchant_category", "is_new_merchant"),
    ("auth_method", "channel"),
    ("auth_method", "device_type"),
    ("auth_method", "cvv_retry_count"),
    ("channel", "device_type"),
    ("card_type", "auth_method"),
    ("card_type", "channel"),
    ("time_of_day_hour", "day_of_week"),
    ("used_vpn", "ip_country_mismatch"),
    ("ip_country_mismatch", "billing_shipping_mismatch"),
    ("cvv_retry_count", "prior_disputes"),
    ("customer_age", "merchant_category"),
    ("is_ai_generated_scam_attempt", "channel"),
    ("is_ai_generated_scam_attempt", "auth_method"),
]


def prepare_base_and_keys(data_dir: Path):
    """Đọc train/test/folds và tạo từ điển khóa TE + DataFrame đặc trưng nền gọn nhẹ (float32/int32)."""
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    folds_df = pd.read_csv(data_dir / "folds.csv")

    assert (train_df["id"].values == folds_df["id"].values).all(), "Thứ tự id trong folds.csv không khớp!"
    folds = folds_df["fold"].values.astype(np.int32)
    y_all = train_df["is_fraud"].values.astype(np.float32)

    n_train = len(train_df)
    full_df = pd.concat(
        [train_df.drop(columns=["id", "is_fraud"]), test_df.drop(columns=["id"])],
        axis=0,
        ignore_index=True,
    )
    del train_df, test_df, folds_df
    gc.collect()

    for c in BOOL_COLS:
        full_df[c] = full_df[c].astype(np.int8)

    key_dict = {}
    for c in CAT_COLS:
        key_dict[c] = pd.factorize(full_df[c])[0].astype(np.int32)
    for c in BOOL_COLS + INT_COLS:
        key_dict[c] = full_df[c].values.astype(np.int32)

    extra_dict = {}
    for c in FLOAT_COLS:
        x = full_df[c].values
        x_100 = np.round(x * 100).astype(np.int32)
        x_10 = np.round(x * 10).astype(np.int32)
        x_int = np.floor(x + 1e-9).astype(np.int32)

        key_dict[f"{c}_exact"] = pd.factorize(x_100)[0].astype(np.int32)
        key_dict[f"{c}_cents"] = (x_100 % 100).astype(np.int32)
        key_dict[f"{c}_tenths"] = ((x_100 // 10) % 10).astype(np.int32)
        key_dict[f"{c}_hund"] = (x_100 % 10).astype(np.int32)
        key_dict[f"{c}_int"] = pd.factorize(x_int)[0].astype(np.int32)
        key_dict[f"{c}_div10"] = pd.factorize(x_int // 10)[0].astype(np.int32)
        key_dict[f"{c}_div100"] = pd.factorize(x_int // 100)[0].astype(np.int32)
        key_dict[f"{c}_div1000"] = pd.factorize(x_int // 1000)[0].astype(np.int32)
        key_dict[f"{c}_10x"] = pd.factorize(x_10)[0].astype(np.int32)
        key_dict[f"{c}_last_d"] = (x_int % 10).astype(np.int32)
        key_dict[f"{c}_last2_d"] = (x_int % 100).astype(np.int32)

        s = full_df[c].astype(str).str.replace(".", "", regex=False).str.lstrip("0")
        key_dict[f"{c}_first1"] = pd.factorize(s.str[:1])[0].astype(np.int32)
        key_dict[f"{c}_first2"] = pd.factorize(s.str[:2])[0].astype(np.int32)
        key_dict[f"{c}_first3"] = pd.factorize(s.str[:3])[0].astype(np.int32)

        extra_dict[f"{c}_cents"] = (x_100 % 100).astype(np.float32)
        extra_dict[f"{c}_tenths"] = ((x_100 // 10) % 10).astype(np.float32)
        extra_dict[f"{c}_hund"] = (x_100 % 10).astype(np.float32)
        extra_dict[f"{c}_is_1dec"] = ((x_100 % 10) == 0).astype(np.float32)

    bool_bitmask = sum(full_df[c].values.astype(np.int32) << i for i, c in enumerate(BOOL_COLS))
    key_dict["bool_bitmask"] = bool_bitmask
    key_dict["risk_flags_count"] = full_df[BOOL_COLS].sum(axis=1).values.astype(np.int32)

    for c1, c2 in PAIR_DEFS:
        key_dict[f"pair_{c1}_{c2}"] = pd.factorize(
            full_df[c1].astype(str) + "_" + full_df[c2].astype(str)
        )[0].astype(np.int32)

    extra_dict["risk_flags_count"] = key_dict["risk_flags_count"].astype(np.float32)
    extra_dict["amount_to_balance"] = (
        full_df["amount_usd"].values / (full_df["account_balance_usd"].values + 1.0)
    ).astype(np.float32)
    extra_dict["merchant_x_velocity"] = (
        full_df["merchant_risk_score"].values * full_df["velocity_score"].values
    ).astype(np.float32)

    # Tạo mảng numeric thuần cho Quantile/Scaler (raw float + int + bool + extra domain/digit nums)
    raw_num_cols = FLOAT_COLS + INT_COLS + BOOL_COLS
    extra_df = pd.DataFrame(extra_dict)
    raw_num_all = np.hstack([full_df[raw_num_cols].values.astype(np.float32), extra_df.values])

    # Tạo DataFrame nền cho GBDT
    base_df = full_df.copy()
    for c in CAT_COLS:
        base_df[c] = base_df[c].astype("category")
    for c in FLOAT_COLS:
        base_df[c] = base_df[c].astype(np.float32)
    for c in INT_COLS:
        base_df[c] = base_df[c].astype(np.int32)
    for k, v in extra_dict.items():
        base_df[k] = v

    del full_df, extra_df
    gc.collect()

    return {
        "n_train": n_train,
        "y_all": y_all,
        "folds": folds,
        "key_dict": key_dict,
        "base_df": base_df,
        "raw_num_all": raw_num_all,
    }


def _compute_single_te_freq(codes_tr_full, y_tr, codes_va, codes_te, n_cats, m=15.0, seed=123):
    """Tính m-estimate TE với inner 5-fold cho tập train ngoài và full outer train cho val/test."""
    gm_full = float(y_tr.mean())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_te = np.zeros(len(y_tr), dtype=np.float32)

    for in_tr, in_va in skf.split(np.zeros(len(y_tr)), y_tr):
        y_in = y_tr[in_tr]
        c_in = codes_tr_full[in_tr]
        gm = float(y_in.mean())
        cnt = np.bincount(c_in, minlength=n_cats)
        pos = np.bincount(c_in, weights=y_in, minlength=n_cats)
        tr_te[in_va] = ((pos + m * gm) / (cnt + m))[codes_tr_full[in_va]]

    cnt_f = np.bincount(codes_tr_full, minlength=n_cats)
    pos_f = np.bincount(codes_tr_full, weights=y_tr, minlength=n_cats)
    te_map = (pos_f + m * gm_full) / (cnt_f + m)
    va_te = te_map[codes_va]
    te_te = te_map[codes_te]

    freq_tr = np.log1p(cnt_f[codes_tr_full]).astype(np.float32)
    freq_va = np.log1p(cnt_f[codes_va]).astype(np.float32)
    freq_te = np.log1p(cnt_f[codes_te]).astype(np.float32)

    return tr_te, va_te, te_te, freq_tr, freq_va, freq_te


def compute_fold_te_features(bundle, fold: int, m: float = 15.0, inner_seed: int = 123):
    """
    Tính toàn bộ ma trận TE logits & Frequency cho outer fold chỉ định.
    Trả về dict chứa các mảng float32 (tr, va, te) cùng tên cột.
    """
    n_train = bundle["n_train"]
    folds = bundle["folds"]
    y_all = bundle["y_all"]
    key_dict = bundle["key_dict"]

    tr_idx = np.where(folds != fold)[0]
    va_idx = np.where(folds == fold)[0]
    te_idx = np.arange(n_train, len(bundle["base_df"]))
    y_tr = y_all[tr_idx]

    te_cols = list(key_dict.keys())
    n_te = len(te_cols)

    X_te_tr = np.zeros((len(tr_idx), n_te), dtype=np.float32)
    X_te_va = np.zeros((len(va_idx), n_te), dtype=np.float32)
    X_te_te = np.zeros((len(te_idx), n_te), dtype=np.float32)

    freq_tr_list, freq_va_list, freq_te_list, freq_names = [], [], [], []

    for i, col in enumerate(te_cols):
        codes = key_dict[col]
        n_cats = int(codes.max()) + 1
        tr_p, va_p, te_p, f_tr, f_va, f_te = _compute_single_te_freq(
            codes[tr_idx], y_tr, codes[va_idx], codes[te_idx], n_cats, m=m, seed=inner_seed
        )
        X_te_tr[:, i] = logit(np.clip(tr_p, 1e-5, 1.0 - 1e-5))
        X_te_va[:, i] = logit(np.clip(va_p, 1e-5, 1.0 - 1e-5))
        X_te_te[:, i] = logit(np.clip(te_p, 1e-5, 1.0 - 1e-5))

        if col.endswith("_exact") or col in ("customer_age", "bool_bitmask"):
            freq_tr_list.append(f_tr)
            freq_va_list.append(f_va)
            freq_te_list.append(f_te)
            freq_names.append(f"freq_{col}")

    F_tr = np.column_stack(freq_tr_list).astype(np.float32)
    F_va = np.column_stack(freq_va_list).astype(np.float32)
    F_te = np.column_stack(freq_te_list).astype(np.float32)

    return {
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "te_idx": te_idx,
        "te_names": [f"te_{c}" for c in te_cols],
        "freq_names": freq_names,
        "X_te_tr": X_te_tr,
        "X_te_va": X_te_va,
        "X_te_te": X_te_te,
        "F_tr": F_tr,
        "F_va": F_va,
        "F_te": F_te,
    }
