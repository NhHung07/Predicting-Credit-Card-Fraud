# Thư mục `configs/`

Chứa các file cấu hình thí nghiệm (định dạng `.yaml` hoặc `.json`).

## Mục đích
Tách biệt **Mã nguồn (Code)** và **Tham số thí nghiệm (Config)**:
- Không viết cứng (hardcode) các thông số như `learning_rate`, `max_depth`, danh sách feature vào trong Notebook hay file Python.
- Dễ dàng tái lập kết quả: nhìn vào file config là biết chính xác thí nghiệm đó đã chạy với thông số gì.
- Dễ dàng so sánh thay đổi trên Git (xem Git diff trực quan khi tăng/giảm tham số).

## Cách đọc config trong Python / Notebook
```python
import yaml

# Đọc file config
with open("configs/lgbm_baseline.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Sử dụng tham số
model_params = cfg["model_params"]
exp_id = cfg["experiment"]["exp_id"]
cat_cols = cfg["features"]["categorical_features"]
```
