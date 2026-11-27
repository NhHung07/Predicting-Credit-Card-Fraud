#!/usr/bin/env python3
"""
Thí nghiệm: PyTorch Wide & Deep Periodic TabResNet (Deep Learning thuần túy - 5-Fold Stratified)
Thành viên: C (Phạm Hồng Nam)
Mã thí nghiệm: mlp_c_v2
Họ mô hình: Deep Learning (Entity Embeddings + Periodic Numerical Embeddings + Wide Linear Skip + ResNet)
100% Không sử dụng LLM.
"""

import copy
import gc
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from shared.fe_digit_te import CAT_COLS, compute_fold_te_features, prepare_base_and_keys

EXP_ID = "mlp_c_v2"
MEMBER = "C"
SEED = 42
N_SPLITS = 5
EPOCHS = 20
BATCH_SIZE = 4096
LR = 6e-4
WEIGHT_DECAY = 2e-2

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "C" / "outputs"
SUB_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PeriodicEmbedding(nn.Module):
    """Nhúng biến liên tục bằng hàm sin/cos đa tần số học được (Gorishniy et al., NeurIPS 2022)."""

    def __init__(self, in_features: int, n_bins: int = 8, sigma: float = 0.05):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(in_features, n_bins) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = 2.0 * np.pi * x.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(1)


class WideDeepPeriodicResNet(nn.Module):
    """Kiến trúc Wide & Deep TabResNet với kết nối tuyến tính skip, Entity Embedding và Periodic Embedding."""

    def __init__(self, emb_dims, n_num: int, n_rest: int, hidden_dim: int = 192, dropout: float = 0.45):
        super().__init__()
        self.cat_embs = nn.ModuleList([nn.Embedding(nc, d) for nc, d in emb_dims])
        self.per = PeriodicEmbedding(n_num, n_bins=8, sigma=0.05)

        total_cont = n_num + n_rest
        in_dim = sum(d for _, d in emb_dims) + (n_num * 16) + total_cont
        self.wide = nn.Linear(total_cont, 1)

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor, n_num: int) -> torch.Tensor:
        xc = x_cont + 0.05 * torch.randn_like(x_cont) if self.training else x_cont
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.cat_embs)]
        x_per = self.per(xc[:, :n_num])
        x = torch.cat(embs + [x_per, xc], dim=1)
        h = self.in_proj(x)
        h = self.act(h + self.block1(h))
        return (self.wide(xc) + self.head(h)).squeeze(1)


@torch.no_grad()
def predict_batched(model, x_cat_np, x_cont_np, n_num: int, batch_size: int = 16384):
    model.eval()
    preds = []
    n = len(x_cat_np)
    for i in range(0, n, batch_size):
        cb = torch.from_numpy(x_cat_np[i : i + batch_size]).to(device)
        xb = torch.from_numpy(x_cont_np[i : i + batch_size]).to(device)
        preds.append(torch.sigmoid(model(cb, xb, n_num)).cpu().numpy())
    return np.concatenate(preds)


def main():
    print(f"=== [START] Huấn luyện {EXP_ID} Deep Learning (Wide&Deep Periodic TabResNet - Member {MEMBER}) ===")
    print(f"Thiết bị tính toán: {device}")
    t0 = time.time()

    bundle = prepare_base_and_keys(DATA_DIR)
    n_train = bundle["n_train"]
    y_all = bundle["y_all"]
    raw_num_all = bundle["raw_num_all"]
    key_dict = bundle["key_dict"]
    n_test = len(bundle["base_df"]) - n_train

    cat_mat = np.column_stack([key_dict[c] for c in CAT_COLS]).astype(np.int64)
    emb_dims = [(int(cat_mat[:, i].max()) + 1, 8) for i in range(cat_mat.shape[1])]

    oof_preds = np.zeros(n_train, dtype=np.float64)
    test_preds = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    for fold in range(N_SPLITS):
        f_start = time.time()
        fe = compute_fold_te_features(bundle, fold, m=15.0, inner_seed=123)
        tr_idx, va_idx, te_idx = fe["tr_idx"], fe["va_idx"], fe["te_idx"]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
        num_tr_qt = qt.fit_transform(raw_num_all[tr_idx]).astype(np.float32)
        num_va_qt = qt.transform(raw_num_all[va_idx]).astype(np.float32)
        num_te_qt = qt.transform(raw_num_all[te_idx]).astype(np.float32)

        rest_tr = np.hstack([fe["X_te_tr"], fe["F_tr"]])
        rest_va = np.hstack([fe["X_te_va"], fe["F_va"]])
        rest_te = np.hstack([fe["X_te_te"], fe["F_te"]])

        sc = StandardScaler()
        rest_tr_s = sc.fit_transform(rest_tr).astype(np.float32)
        rest_va_s = sc.transform(rest_va).astype(np.float32)
        rest_te_s = sc.transform(rest_te).astype(np.float32)

        cont_tr = np.hstack([num_tr_qt, rest_tr_s]).astype(np.float32)
        cont_va = np.hstack([num_va_qt, rest_va_s]).astype(np.float32)
        cont_te = np.hstack([num_te_qt, rest_te_s]).astype(np.float32)

        n_num = num_tr_qt.shape[1]
        n_rest = rest_tr_s.shape[1]

        torch.manual_seed(SEED + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold)

        tr_ds = TensorDataset(
            torch.from_numpy(cat_mat[tr_idx]),
            torch.from_numpy(cont_tr),
            torch.from_numpy(y_tr),
        )
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        model = WideDeepPeriodicResNet(emb_dims, n_num, n_rest).to(device)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_va_preds = None
        best_state = None
        best_ep = 0

        for ep in range(EPOCHS):
            model.train()
            for cb, xb, yb in tr_loader:
                cb = cb.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad()
                loss = crit(model(cb, xb, n_num), yb)
                loss.backward()
                opt.step()
            sched.step()

            p_va = predict_batched(model, cat_mat[va_idx], cont_va, n_num)
            auc = roc_auc_score(y_va, p_va)
            if auc > best_auc:
                best_auc = auc
                best_va_preds = p_va
                best_state = copy.deepcopy(model.state_dict())
                best_ep = ep + 1

        model.load_state_dict(best_state)
        p_te = predict_batched(model, cat_mat[te_idx], cont_te, n_num)

        oof_preds[va_idx] = best_va_preds
        test_preds += p_te / N_SPLITS
        fold_aucs.append(best_auc)
        print(
            f"--- Fold {fold+1}/{N_SPLITS} | Best AUC: {best_auc:.5f} (Ep {best_ep}) | "
            f"Time: {time.time() - f_start:.1f}s ---"
        )

        del fe, tr_ds, tr_loader, cont_tr, cont_va, cont_te, model, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    overall_auc = roc_auc_score(y_all, oof_preds)
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    print("=" * 60)
    print(f"HOÀN THÀNH {EXP_ID} (Deep Learning) | Tổng thời gian: {(time.time() - t0) / 60.0:.2f} phút")
    print(f"AUC từng fold: {[round(x, 5) for x in fold_aucs]}")
    print(f"Mean Fold AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
    print(f">> OVERALL OOF AUC: {overall_auc:.5f} <<")
    print("=" * 60)

    np.save(OUTPUT_DIR / f"{EXP_ID}_oof.npy", oof_preds)
    np.save(OUTPUT_DIR / f"{EXP_ID}_test.npy", test_preds)

    sub_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub_df["is_fraud"] = test_preds
    sub_df.to_csv(SUB_DIR / f"{EXP_ID}.csv", index=False)
    print(f"Đã lưu OOF, Test và Submission: {SUB_DIR / f'{EXP_ID}.csv'}")


if __name__ == "__main__":
    main()
