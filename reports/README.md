# Thư mục `src/`, `configs/`, `scripts/`, `reports/`

Giải thích vai trò của các thư mục hỗ trợ trong dự án:

- **`src/`** — Chứa các hàm/module Python dùng chung khi nhóm muốn tách phần xử lý dữ liệu, tạo đặc trưng (`features.py`), hoặc hàm đánh giá CV ra khỏi Notebook để cả 3 thành viên cùng import (có thể `git clone` repo ngay trong Kaggle Notebook để gọi `from src... import ...`).
  - **`src/models/`** — Nơi đặt các hàm huấn luyện theo từng dòng mô hình (LightGBM, XGBoost, CatBoost, Linear/MLP).
- **`configs/`** — Chứa các file cấu hình `.yaml` hoặc `.json` lưu bộ tham số (hyperparameters) tốt nhất sau khi tune bằng Optuna và danh sách các feature được chọn.
- **`scripts/`** — Chứa các script tiện ích chạy trên máy local (ví dụ script tải dữ liệu hoặc nộp bài qua Kaggle CLI).
- **`reports/`** — Chứa sản phẩm nộp môn Học máy:
  - `reports/figures/`: Biểu đồ xuất ra từ EDA, Feature Importance, đường cong ROC, so sánh các mô hình.
  - `reports/slides/`: Slide thuyết trình của nhóm.
  - File báo cáo tổng kết bài tập lớn.
