#!/usr/bin/env bash
set -e

# Thư mục đích
TARGET_DIR="data/raw"
COMPETITION="playground-series-s6e11"

mkdir -p "$TARGET_DIR"

echo "==> Đang tải dữ liệu cuộc thi $COMPETITION vào $TARGET_DIR..."
/home/hung/miniconda3/envs/ai_env/bin/kaggle competitions download -c "$COMPETITION" -p "$TARGET_DIR"

echo "==> Đang giải nén dữ liệu..."
unzip -o "$TARGET_DIR/${COMPETITION}.zip" -d "$TARGET_DIR"

echo "==> Hoàn tất! Danh sách file trong $TARGET_DIR:"
ls -lh "$TARGET_DIR"
