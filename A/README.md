# Thư mục của A (Trần Nhật Hưng)

- **Nhánh Git:** `a-hung`
- **Phạm vi phụ trách:** Baseline, hệ **LightGBM** + Feature Engineering v1/v2 + tuning Optuna cho LightGBM.
- **Quy ước đặt tên `exp_id`:** `lgbm_a_v1`, `lgbm_a_v2`, ...

## Cấu trúc bên trong `A/`

```
A/
├── README.md
├── 02_baseline_lgbm.ipynb     # Baseline LightGBM 5-fold (CV 0.94544, LB 0.94568)
├── lgbm_baseline.yaml         # Tham số của lgbm_a_v1
├── lgbm_a_v2_fe.ipynb         # (gợi ý) Thử nghiệm FE tiếp theo
└── outputs/                   # (không commit) <exp_id>_oof.npy, <exp_id>_test.npy, fe_*.parquet
```

## Quy trình làm việc

1. Đọc dữ liệu từ `ROOT / "data"` (`train.csv`, `test.csv`, `folds.csv`) với:
   ```python
   ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
   ```
2. Lưu OOF (`shape=(699960,)`) và test prediction (`shape=(299984,)`) vào `A/outputs/<exp_id>_oof.npy` và `A/outputs/<exp_id>_test.npy` (hoặc lên Kaggle Dataset chung `s6e11-team-artifacts`).
3. Lưu file nộp vào `submissions/<exp_id>.csv`.
4. Chạy **Restart & Run All** để lưu kèm đồ thị/output trong `.ipynb` rồi ghi 1 dòng vào `results/results.csv`.
