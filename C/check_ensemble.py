import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

y = pd.read_csv("data/train.csv")["is_fraud"].values

files = {
    "v1": "C/outputs/cat_c_v1_oof.npy",
    "v3_D5": "C/outputs/cat_c_v3_d5_oof.npy",
    "v3_D6": "C/outputs/cat_c_v3_d6_oof.npy",
    "v3_blend": "C/outputs/cat_c_v3_blend_oof.npy",
}

preds = {}

print("=== INDIVIDUAL OOF AUC ===")
for name, path in files.items():
    p = np.load(path).reshape(-1)
    assert len(p) == len(y), f"Length mismatch: {name}"
    preds[name] = p
    print(f"{name:12s}: {roc_auc_score(y, p):.6f}")

print("\n=== CORRELATION ===")
print(pd.DataFrame(preds).corr().round(5))

print("\n=== BLEND SEARCH ===")
p1 = preds["v1"]
p3 = preds["v3_blend"]

for w in np.arange(0, 1.01, 0.05):
    blend = w*p1 + (1-w)*p3
    auc = roc_auc_score(y, blend)
    print(f"v1={w:.2f}, v3={1-w:.2f}: AUC={auc:.6f}")
