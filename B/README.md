# Thư mục của B (Cao Anh Minh)

- **Nhánh Git:** `b-minh`
- **Phạm vi phụ trách:** Hệ **XGBoost** (và/hoặc **CatBoost**) trên cùng `data/folds.csv`, thử nghiệm các biến tương tác mới (tỷ lệ giao dịch, lịch sử rủi ro, OOF target encoding) + tuning Optuna.
- **Quy ước đặt tên `exp_id`:** `xgb_b_v1`, `xgb_b_v2`, `cat_b_v1`, ...

## Cấu trúc đề xuất trong `B/`

```
B/
├── README.md
├── 01_xgb_baseline.ipynb      # Baseline XGBoost 5-fold dùng chung data/folds.csv
├── xgb_baseline.yaml          # (tuỳ chọn) Lưu bộ tham số chuẩn
├── 02_xgb_fe.ipynb            # Thử nghiệm Feature Engineering & Optuna
└── outputs/                   # (không commit) xgb_b_v1_oof.npy, xgb_b_v1_test.npy, ...
```

## Quy trình làm việc

1. Đọc dữ liệu từ `ROOT / "data"` và bắt buộc dùng `data/folds.csv` (5-fold Stratified, `seed=42`) để điểm CV so sánh trực tiếp được với A và C.
   ```python
   ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
   ```
2. Xuất đủ `B/outputs/<exp_id>_oof.npy` (`(699960,)`) và `B/outputs/<exp_id>_test.npy` (`(299984,)`) đúng thứ tự `train.csv`/`test.csv`, đồng thời lưu file nộp vào `submissions/<exp_id>.csv`.
3. Chia sẻ file `.npy` quan trọng lên Kaggle Dataset `s6e11-team-artifacts` để C ghép Ensemble.
4. Chạy **Restart & Run All** trước khi commit notebook và cập nhật 1 dòng vào `results/results.csv`.
