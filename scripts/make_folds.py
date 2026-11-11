#!/usr/bin/env python3
"""
Tạo file folds.csv dùng chung cho cả nhóm.
Phương pháp: StratifiedKFold (5 folds, shuffle=True, random_state=42)
Đầu ra: data/processed/folds.csv (gồm 2 cột: id, fold)
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RAW_TRAIN_PATH = Path("data/raw/train.csv")
OUTPUT_PATH = Path("data/processed/folds.csv")
N_SPLITS = 5
SEED = 42

def main():
    if not RAW_TRAIN_PATH.exists():
        print(f"Lỗi: Không tìm thấy {RAW_TRAIN_PATH}")
        sys.exit(1)

    print(f"==> Đang đọc {RAW_TRAIN_PATH} (cột 'id' và 'is_fraud')...")
    df = pd.read_csv(RAW_TRAIN_PATH, usecols=["id", "is_fraud"])
    n_samples = len(df)
    n_fraud = df["is_fraud"].sum()
    fraud_rate = df["is_fraud"].mean()

    print(f"Tổng số mẫu: {n_samples:,}")
    print(f"Số mẫu gian lận (is_fraud=1): {int(n_fraud):,} ({fraud_rate:.4%})")

    print(f"==> Đang chia {N_SPLITS}-fold Stratified (seed={SEED})...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["is_fraud"])):
        df.loc[val_idx, "fold"] = fold

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[["id", "fold"]]
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"==> Đã lưu file: {OUTPUT_PATH}")

    print("\nKiểm tra phân bố giữa các fold:")
    summary = df.groupby("fold")["is_fraud"].agg(
        total_samples="count",
        fraud_samples="sum",
        fraud_rate="mean"
    )
    print(summary)

if __name__ == "__main__":
    main()
