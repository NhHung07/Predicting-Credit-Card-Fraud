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

> Lộ trình: [`shared/timeline.md`](shared/timeline.md)
> Hướng dẫn dùng Kaggle: [`shared/kaggle_guide.md`](shared/kaggle_guide.md)

## Cấu trúc thư mục

```
Predicting-Credit-Card-Fraud/
├── README.md
├── results/
│   └── results.csv      # (commit) nhật ký MỌI thí nghiệm của cả nhóm
├── requirements.txt     # thư viện (pip)
├── environment.yml      # môi trường conda
├── data/                # (không commit) train.csv, test.csv, sample_submission.csv, folds.csv
├── shared/              # dùng chung cả nhóm
│   ├── eda.ipynb        #   EDA
│   ├── make_folds.py    #   tạo data/folds.csv
│   ├── download_data.sh #   tải dữ liệu từ Kaggle
│   ├── timeline.md
│   └── kaggle_guide.md
├── A/                   # Hưng: notebook + config + outputs/ + ensemble
├── B/                   # Minh: notebook + config + outputs/
├── C/                   # Nam: notebook + config + outputs/
└── submissions/         # (không commit) file nộp Kaggle: <exp_id>.csv
```

Trong thư mục của mỗi thành viên (`A/`, `B/`, `C/`):
- `<tên>.ipynb`, `<tên>.yaml`: notebook và tham số của từng thí nghiệm (mỗi thành viên chỉ sửa trong thư mục của mình).
- `outputs/<exp_id>_oof.npy`: dự đoán out-of-fold, `shape = (699960,)`, đúng thứ tự `train.csv`.
- `outputs/<exp_id>_test.npy`: dự đoán trên test (trung bình các fold), `shape = (299984,)`, đúng thứ tự `test.csv`.
- Dữ liệu `data/` để phẳng 4 file dùng chung (`train.csv`, `test.csv`, `sample_submission.csv`, `folds.csv`), không tách `raw/` và `processed/`; nếu thành viên tạo bộ đặc trưng riêng (`.parquet`), lưu vào `<A|B|C>/outputs/` để tránh ghi đè lẫn nhau.

Notebook nên tìm thư mục gốc repo bằng đoạn sau để chạy được cả từ gốc repo lẫn từ trong `A/`, `B/`, `C/`:

```python
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
DATA_DIR = ROOT / "data"
```

## Cài đặt môi trường (conda)

```bash
# Cách 1: dùng env có sẵn rồi cài thêm thư viện
conda activate ai_env
pip install -r requirements.txt

# Cách 2: tạo env mới cho project
conda env create -f environment.yml -n fraud
conda activate fraud
```

Trên **Kaggle Notebook** đã có sẵn gần hết thư viện (lightgbm, xgboost, catboost, sklearn...), không cần cài.

## Tải dữ liệu (máy local)

```bash
# Cần ~/.kaggle/kaggle.json (xem shared/kaggle_guide.md)
kaggle competitions download -c playground-series-s6e11 -p data
unzip -o data/playground-series-s6e11.zip -d data
python shared/make_folds.py   # tạo data/folds.csv
```

## Quy ước chung của nhóm

1. **Fold dùng chung:** cả nhóm dùng chung một file `folds.csv` (StratifiedKFold, `n_splits=5`, `shuffle=True`, `random_state=42`).
   Một người tạo file này, sau đó upload lên Kaggle Dataset để cả nhóm dùng. **Không tự chia fold riêng**, vì CV phải so sánh được với nhau.
2. **Luôn lưu OOF + test prediction** cho mọi thí nghiệm, nếu không sẽ không ensemble được.
3. **Tên thí nghiệm** `exp_id` = `<model>_<mã thành viên>_v<số>`, ví dụ `lgbm_a_v3`. Dùng cùng một tên cho file OOF, file pred, file submission và dòng trong `results/results.csv`.
4. **Ghi kết quả** của mọi thí nghiệm (kể cả thất bại) vào [`results/results.csv`](results/results.csv).
5. **Tin vào CV hơn Public LB.** Public LB chỉ tính trên một phần nhỏ của tập test.
6. **Không commit:** `data/`, `outputs/`, `submissions/`, `kaggle.json`.
   Notebook **được commit kèm output** để xem đồ thị ngay trên GitHub. Trước khi commit hãy **Restart & Run All** để output sạch, đúng thứ tự, và tránh in bảng/log quá dài.
7. **Kiểm tra file nộp:** không có NaN/Inf, giá trị là xác suất trong [0, 1] (không làm tròn thành 0/1), đủ 299,984 dòng.

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
| _TODO_ | Baseline | | | |
| _TODO_ | Submission cuối #1 | | | |
| _TODO_ | Submission cuối #2 | | | |
