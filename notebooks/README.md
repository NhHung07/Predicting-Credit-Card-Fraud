# Thư mục `notebooks/`

Chứa các Jupyter Notebook phục vụ khám phá dữ liệu, thử nghiệm và chạy trên Kaggle.

## Phân chia thư mục

- `notebooks/kaggle/` — Các notebook **chính thức** đã chạy ổn định trên Kaggle (EDA hoàn chỉnh, các mô hình chính của từng thành viên, notebook Ensemble cuối cùng). Đặt tên theo thứ tự hoặc theo `exp_id`, ví dụ:
  - `01_eda.ipynb`
  - `02_make_folds.ipynb`
  - `lgbm_a_v1.ipynb`
  - `xgb_b_v1.ipynb`
  - `ensemble_c_v1.ipynb`
- `notebooks/scratch/A/`, `B/`, `C/` — Không gian nháp riêng của từng thành viên để thử nghiệm nhanh, không yêu cầu trình bày đẹp.

## Lưu ý khi commit Notebook lên GitHub

- Đã cấu hình `nbstripout` trong `requirements.txt` để tự động xóa phần output nặng (bảng dài, log hàng nghìn dòng) trước khi commit, giúp tránh xung đột Git và giữ repo nhẹ.
- Nếu muốn giữ lại biểu đồ phục vụ báo cáo, hãy xuất ảnh ra `reports/figures/` bằng `plt.savefig(...)`.
