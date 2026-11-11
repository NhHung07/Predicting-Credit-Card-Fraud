# Thư mục `outputs/`

> Toàn bộ file kết quả trong `outputs/` đều được **gitignore**. Khi chạy trên Kaggle Notebook, hãy lưu các file OOF và Test Prediction quan trọng vào **Kaggle Dataset nội bộ của nhóm** để Thành viên C tổng hợp làm Ensemble.

## Cấu trúc & Quy ước đặt tên theo `<exp_id>`

Với mỗi thí nghiệm có mã `<exp_id>` (ví dụ `lgbm_a_v2`):

- `outputs/oof/<exp_id>_oof.npy` — Mảng 1 chiều (`shape = (699960,)`, thứ tự khớp đúng với `train.csv` gốc) chứa xác suất dự đoán out-of-fold.
- `outputs/preds/<exp_id>_test.npy` — Mảng 1 chiều (`shape = (299984,)`, thứ tự khớp đúng với `test.csv` gốc) chứa xác suất dự đoán trung bình qua các fold trên tập test.
- `outputs/models/` — Trọng số model đã huấn luyện (nếu cần lưu lại).
- `outputs/submissions/<exp_id>.csv` — File CSV nộp lên Kaggle, gồm đúng 2 cột `id,is_fraud`.

## Kiểm tra nhanh trước khi lưu / nộp

1. Không có giá trị `NaN` hoặc `Inf` trong `oof` và `test`.
2. Giá trị dự đoán là **xác suất liên tục** nằm trong `[0, 1]` (tuyệt đối không làm tròn thành `0`/`1` vì metric là **ROC AUC**).
3. Số dòng của file submission bằng đúng `sample_submission.csv` (`299,984` dòng dự đoán + `1` dòng header = `299,985` dòng).
