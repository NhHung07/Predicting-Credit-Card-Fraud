# Thư mục `data/`

> Toàn bộ file dữ liệu trong `data/` đều được **gitignore** (trừ `README.md` và `.gitkeep`). Không commit file `.csv`, `.parquet`, `.zip` lên GitHub.

## Cấu trúc

- `data/raw/` — Dữ liệu gốc tải từ Kaggle (`train.csv`, `test.csv`, `sample_submission.csv`).
- `data/processed/` — Dữ liệu đã qua xử lý và file chia fold dùng chung:
  - `folds.csv`: gồm 2 cột `id,fold` (chia bằng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trên cột `is_fraud`).
  - Các bảng đặc trưng đã xử lý (nên lưu định dạng `.parquet` để đọc nhanh và giữ nguyên kiểu dữ liệu).

## Cách lấy dữ liệu

- **Trên Kaggle Notebook:** gắn trực tiếp Competition Dataset (`/kaggle/input/playground-series-s6e11/`) và Team Private Dataset chứa `folds.csv`.
- **Trên máy local:**
  ```bash
  kaggle competitions download -c playground-series-s6e11 -p data/raw
  unzip -o data/raw/playground-series-s6e11.zip -d data/raw
  ```

## Thông tin các trường dữ liệu (`train.csv`: 699,960 dòng × 26 cột, `test.csv`: 299,984 dòng × 25 cột)

- **Định danh & Nhãn:** `id`, `is_fraud` *(chỉ có ở train, giá trị 0/1)*.
- **Số tiền & Số dư:** `amount_usd`, `account_balance_usd`.
- **Thời gian & Tần suất:** `hours_since_last_txn`, `txn_count_last_24h`, `time_of_day_hour`, `day_of_week`.
- **Khách hàng & Thẻ:** `distance_from_home_km`, `card_age_months`, `customer_age`.
- **Điểm rủi ro & Lịch sử:** `cvv_retry_count`, `velocity_score`, `merchant_risk_score`, `prior_disputes`.
- **Phân loại (Categorical):** `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type`.
- **Cờ nhị phân (Boolean/Flag):** `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`.
