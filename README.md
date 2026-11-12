# Predicting Credit Card Fraud – Kaggle Playground S6E11

Bài tập lớn môn **Học máy**. Nhóm tham gia cuộc thi Kaggle
[Playground Series S6E11 – Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11).

| | |
|---|---|
| **Bài toán** | Phân loại nhị phân: dự đoán **xác suất** giao dịch thẻ là gian lận (`is_fraud`) |
| **Metric** | ROC AUC (nộp xác suất trong khoảng [0, 1], không nộp nhãn 0/1) |
| **Dữ liệu** | `train.csv` (~700k dòng), `test.csv` (~300k dòng), `sample_submission.csv` |
| **Hạn chót** | **30/11/2026 23:59 UTC** (06:59 sáng 01/12 giờ VN) |

## Thành viên

| Mã | Họ tên | Nhánh Git |
|---|---|---|
| A | _Trần Nhật Hưng_ | `a-hung` |
| B | _Cao Anh Minh_ | `b-minh` | 
| C | _Phạm Hồng Nam_ | `c-nam` |

> Lộ trình và phân công chi tiết: [`docs/timeline.md`](docs/timeline.md)
> Hướng dẫn dùng Kaggle: [`docs/kaggle_guide.md`](docs/kaggle_guide.md)

## Cấu trúc thư mục

```
Predicting-Credit-Card-Fraud/
├── README.md
├── requirements.txt          # Danh sách thư viện (pip)
├── environment.yml           # Môi trường conda
├── configs/                  # File YAML cấu hình từng thí nghiệm (params, features)
├── data/                     # (không commit) dữ liệu
│   ├── raw/                  #   train.csv, test.csv, sample_submission.csv
│   └── processed/            #   feature đã xử lý, folds.csv
├── docs/                     # Tài liệu nội bộ của nhóm
├── experiments/
│   └── results.csv           # (commit) nhật ký MỌI thí nghiệm
├── notebooks/
│   ├── kaggle/               # Notebook chạy trên Kaggle (bản "chính thức")
│   └── scratch/{A,B,C}/      # Notebook nháp cá nhân
├── outputs/                  # (không commit) kết quả train
│   ├── oof/                  #   dự đoán out-of-fold:  <exp_id>_oof.npy
│   ├── preds/                #   dự đoán trên test:    <exp_id>_test.npy
│   ├── models/               #   model đã lưu
│   └── submissions/          #   file nộp Kaggle:      <exp_id>.csv
├── reports/                  # Báo cáo, slide, hình ảnh
├── scripts/                  # Shell script tiện ích (tải data, submit)
└── src/                      # Code Python dùng chung (module hóa từ notebook)
    └── models/
```

Mỗi thư mục có `README.md` riêng mô tả nội dung và quy ước đặt tên.

## Cài đặt môi trường (conda)

```bash
# Cách 1: dùng env có sẵn rồi cài thêm thư viện
conda activate ai_env
pip install -r requirements.txt

# Cách 2: tạo env mới cho project
conda env create -f environment.yml -n fraud
conda activate fraud

# Xóa output notebook trước khi commit (chạy 1 lần sau khi clone)
nbstripout --install
```

Trên **Kaggle Notebook** đã có sẵn gần hết thư viện (lightgbm, xgboost, catboost, sklearn...), không cần cài.

## Tải dữ liệu (máy local)

```bash
# Cần ~/.kaggle/kaggle.json (xem docs/kaggle_guide.md)
kaggle competitions download -c playground-series-s6e11 -p data/raw
unzip -o data/raw/playground-series-s6e11.zip -d data/raw
```

## Quy ước chung của nhóm

1. **Fold dùng chung:** cả nhóm dùng chung một file `folds.csv` (StratifiedKFold, `n_splits=5`, `shuffle=True`, `random_state=42`).
   Một người tạo file này, sau đó upload lên Kaggle Dataset để cả nhóm dùng. **Không tự chia fold riêng**, vì CV phải so sánh được với nhau.
2. **Luôn lưu OOF + test prediction** cho mọi thí nghiệm, nếu không sẽ không ensemble được.
3. **Tên thí nghiệm** `exp_id` = `<model>_<mã thành viên>_v<số>`, ví dụ `lgbm_a_v3`. Dùng cùng một tên cho file OOF, file pred, file submission và dòng trong `experiments/results.csv`.
4. **Ghi kết quả** của mọi thí nghiệm (kể cả thất bại) vào [`experiments/results.csv`](experiments/results.csv).
5. **Tin vào CV hơn Public LB.** Public LB chỉ tính trên một phần nhỏ của tập test.
6. **Không commit:** `data/`, `outputs/`, `kaggle.json`, notebook còn output.

## Git workflow

- Mỗi thành viên làm việc trên **nhánh riêng theo tên** (`a-<ten>`, `b-<ten>`, `c-<ten>`).
- `main` chỉ chứa code đã chạy được. Đưa code vào `main` bằng **Pull Request**.
- Trước khi bắt đầu làm, cập nhật nhánh của mình từ `main`:

```bash
git checkout b-<ten>
git fetch origin
git merge origin/main          # hoặc: git rebase origin/main
# ... làm việc ...
git add -A
git commit -m "exp: xgb_b_v2 cv=0.9xxx"
git push origin b-<ten>        # rồi mở PR vào main trên GitHub
```

Gợi ý tiền tố commit: `feat:` (thêm chức năng), `exp:` (thí nghiệm), `fix:`, `docs:`.

## Kết quả

| exp_id | Mô tả | CV AUC | Public LB | Private LB |
|---|---|---|---|---|
| `lgbm_a_v1` | LightGBM Baseline (24 raw features, 5-fold Stratified) | **0.94544** ± 0.00321 | **0.94568** | _[Chưa mở]_ |
| _TODO_ | Submission cuối #1 | | | |
| _TODO_ | Submission cuối #2 | | | |
