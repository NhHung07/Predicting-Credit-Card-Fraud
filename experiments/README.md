# Thư mục `experiments/`

Chứa file [`results.csv`](results.csv) — nhật ký ghi lại **mọi thí nghiệm** của cả 3 thành viên.

## Quy ước ghi `results.csv`

Mỗi khi chạy xong 1 thí nghiệm (dù tăng hay giảm CV), thêm 1 dòng vào `results.csv`:

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `date` | Ngày chạy (`YYYY-MM-DD`) | `2026-11-15` |
| `exp_id` | Mã thí nghiệm (`<model>_<member>_v<số>`) | `lgbm_a_v1` |
| `member` | Thành viên thực hiện (`A`, `B`, `C`) | `A` |
| `model` | Tên thuật toán | `LightGBM` |
| `features` | Tập đặc trưng sử dụng | `raw_24f` hoặc `fe_v2_38f` |
| `n_folds` | Số fold CV | `5` |
| `cv_auc` | Điểm ROC AUC trên toàn bộ OOF (hoặc trung bình 5 fold) | `0.94125` |
| `cv_std` | Độ lệch chuẩn AUC giữa 5 fold | `0.0018` |
| `lb_public` | Điểm Public Leaderboard (để trống nếu không submit) | `0.94080` |
| `notebook_or_commit` | Tên notebook hoặc mã commit tương ứng | `notebooks/kaggle/lgbm_a_v1.ipynb` |
| `notes` | Ghi chú ngắn gọn thay đổi chính so với bản trước | `baseline default params` |

> File `.gitattributes` ở thư mục gốc đã bật chế độ `merge=union` cho `experiments/results.csv`, nên nhiều thành viên có thể thêm dòng mới trên các nhánh khác nhau và merge vào `main` mà không bị xung đột.
