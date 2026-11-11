"""Transformer"""
import hashlib
import json
from pathlib import Path

import pandas as pd

SPLIT_SEED = 2026
TRAIN_PER_MILLE = 100    
VAL_PER_MILLE   = 100    

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "subset"; OUT.mkdir(exist_ok=True)

def in_subset(row_id: str, per_mille: int) -> bool:
    key = f"{SPLIT_SEED}|modeling-subset|{row_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) % 1000 < per_mille

def make(src, dst, per_mille):
    kept, total = [], 0
    for ch in pd.read_csv(PROC / src, dtype=str, keep_default_na=False,
                          na_values=[""], chunksize=500_000):
        total += len(ch)
        kept.append(ch[ch["id"].map(lambda i: in_subset(i, per_mille))])
    df = pd.concat(kept, ignore_index=True)
    df.to_csv(OUT / dst, index=False)
    print(f"{src}: {total:,} -> {dst}: {len(df):,}")
    return df

stats = {}
tr = make("train_supervised.csv", "train_sub.csv", TRAIN_PER_MILLE)
va = make("validation.csv", "val_sub.csv", VAL_PER_MILLE)
ps = make("train_pseudo_unlabeled.csv", "pseudo_sub.csv", TRAIN_PER_MILLE)   
for name, df in (("train_sub", tr), ("val_sub", va)):
    stats[name] = {"rows": len(df),
                   "recommendation": df["recommendation_status"].value_counts(dropna=False).to_dict(),
                   "rating_band": df["rating_band"].value_counts(dropna=False).to_dict()}
stats["pseudo_sub"] = {"rows": len(ps)}
stats["rule"] = "sha256(f'{SPLIT_SEED}|modeling-subset|{id}') mod 1000 < per_mille"
stats["per_mille"] = {"train": TRAIN_PER_MILLE, "val": VAL_PER_MILLE}
(ROOT / "reports" / "modeling_subset.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2, default=int), encoding="utf-8")