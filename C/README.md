# Thư mục của C (Phạm Hồng Nam)

- **Nhánh Git:** `c-nam`
- **Phạm vi phụ trách:** Mô hình đa dạng (**CatBoost** / **Neural Net / MLP** / **Logistic/Ridge**) để tạo độ đa dạng cho tập dự đoán, và phụ trách **Ensemble / Stacking** tổng hợp OOF của cả 3 thành viên (A, B, C) ở Giai đoạn 3.
- **Quy ước đặt tên `exp_id`:** `cat_c_v1`, `mlp_c_v1`, `ens_c_v1`, ...

## Cấu trúc đề xuất trong `C/`

```
C/
├── README.md
├── 01_catboost.ipynb          # Mô hình đơn lẻ khác họ (CatBoost / MLP) trên data/folds.csv
├── 02_ensemble_stacking.ipynb # Gom *_oof.npy và *_test.npy từ A/, B/, C/ để blend/stacking
└── outputs/                   # (không commit) cat_c_v1_oof.npy, ens_c_v1_oof.npy, ...
```

## Quy trình làm việc

1. Đọc dữ liệu từ `ROOT / "data"` và bắt buộc dùng `data/folds.csv` (5-fold Stratified, `seed=42`).
   ```python
   ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
   ```
2. Ở Giai đoạn 2: phát triển model khác họ với LightGBM/XGBoost (như CatBoost, Neural Net/MLP), lưu `C/outputs/<exp_id>_oof.npy` và `C/outputs/<exp_id>_test.npy`.
3. Ở Giai đoạn 3: đọc toàn bộ các file `<exp_id>_oof.npy` và `<exp_id>_test.npy` từ `A/outputs/`, `B/outputs/`, `C/outputs/` (hoặc từ Kaggle Dataset `s6e11-team-artifacts`), tối ưu trọng số Blend / Rank Average / Stacking trên OOF theo `roc_auc_score`, xuất file chốt vào `submissions/ens_c_v*.csv`.
4. Chạy **Restart & Run All** trước khi commit notebook và cập nhật kết quả vào `results/results.csv`.
