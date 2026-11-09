import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from text_norm import normalize

SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
BAND = {"rate_min": 0.0, "rate_max": 5.0, "low_max": 2.5, "high_min": 4.0}
BAND_CODE = {"low": 0, "medium": 1, "high": 2}

LABEL_MAP = {
    "recommended": "recommended",
    "not_recommended": "not_recommended",
    "no_idea": "no_idea",
    "nan": None,
}

ROOT = Path(__file__).resolve().parent.parent
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"
PROC.mkdir(parents=True, exist_ok=True); REP.mkdir(parents=True, exist_ok=True)
NORM = PROC / "normalized.parquet"
q = lambda p: Path(p).as_posix()
t0 = time.time()
def log(m): print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)

def stable_hash(row_id: str) -> int:              
    key = f"{MASK_SEED}|{row_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)

con = duckdb.connect()
con.execute("SET memory_limit='10GB';")
con.execute(f"SET threads={os.cpu_count()};")
con.execute(f"SET temp_directory='{q(PROC / 'duck_tmp')}';")
SRC = f"read_parquet('{q(PROC / 'comments.parquet')}')"

vocab = con.execute(f"""
  SELECT trim(recommendation_status) AS v, count(*) AS n FROM {SRC}
  WHERE recommendation_status IS NOT NULL 
    AND trim(recommendation_status) <> ''
    AND lower(trim(recommendation_status)) NOT IN ('nan', 'null', 'none')
  GROUP BY 1 ORDER BY n DESC""").df()
unknown = vocab[~vocab["v"].isin(LABEL_MAP)]
if len(unknown):
    print(vocab)
    raise SystemExit(f"\nUnknown values not found in LABEL_MAP:\n{unknown}\nComplete the mapping and run again.")
con.register("label_map", pd.DataFrame({"raw": list(LABEL_MAP), "label": list(LABEL_MAP.values())}))

con.execute(f"""CREATE OR REPLACE VIEW stage AS
  SELECT id, product_id, body, trim(recommendation_status) AS label_raw,
         rate AS rate_raw, TRY_CAST(rate AS DOUBLE) AS rate_num FROM {SRC}""")
ex = con.execute("""
  SELECT count(*) AS n_total,
    count(*) FILTER (WHERE id IS NULL OR trim(id)='') AS id_missing,
    count(*) FILTER (WHERE product_id IS NULL OR trim(product_id)='') AS product_id_missing,
    count(*) FILTER (WHERE body IS NULL OR trim(body)='') AS body_empty,
    count(*) - count(DISTINCT id) AS duplicate_id_extra_rows
  FROM stage""").df().iloc[0].astype(int).to_dict()
log(f"quality gate: {ex}")

con.execute("""CREATE OR REPLACE VIEW stage_ok AS SELECT * FROM stage
  WHERE id IS NOT NULL AND trim(id)<>'' AND product_id IS NOT NULL AND trim(product_id)<>''
    AND body IS NOT NULL AND trim(body)<>''""")
if ex["duplicate_id_extra_rows"] > 0:   
    dedup = """SELECT * EXCLUDE (rn) FROM (SELECT *, row_number() OVER
      (PARTITION BY id ORDER BY hash(body), product_id) AS rn FROM stage_ok) WHERE rn = 1"""
else:
    dedup = "SELECT * FROM stage_ok"
con.execute(f"CREATE OR REPLACE VIEW valid AS {dedup}")
rate_diag = con.execute("""
  SELECT count(*) FILTER (WHERE rate_raw IS NULL OR trim(rate_raw)='') AS rate_missing,
         count(*) FILTER (WHERE trim(rate_raw)<>'' AND rate_num IS NULL) AS rate_non_numeric,
         count(*) FILTER (WHERE rate_num < 0 OR rate_num > 5) AS rate_out_of_range,
         count(*) AS n_valid_rows FROM valid""").df().iloc[0].astype(int).to_dict()
log(f"rate diagnostics: {rate_diag}")

sql = f"""
SELECT v.id, v.product_id, m.label AS label,
  CASE 
       WHEN v.rate_num IS NULL OR v.rate_num <= {BAND['rate_min']} OR v.rate_num > {BAND['rate_max']} THEN NULL
       WHEN v.rate_num < {BAND['low_max']}  THEN 'low'
       WHEN v.rate_num < {BAND['high_min']} THEN 'medium'
       ELSE 'high' END AS rating_band,
  v.body AS body_raw
FROM valid v LEFT JOIN label_map m ON v.label_raw = m.raw"""

COLS = ["id", "product_id", "label", "rating_band", "body"]
SCHEMA = pa.schema([(c, pa.string()) for c in COLS])
writer, n_written, n_empty_after_norm = pq.ParquetWriter(q(NORM), SCHEMA, compression="zstd"), 0, 0
for i, batch in enumerate(con.sql(sql).to_arrow_reader(250_000)):
    df = batch.to_pandas()
    df["body"] = df["body_raw"].map(normalize)
    keep = df["body"] != ""
    n_empty_after_norm += int((~keep).sum())
    out = df.loc[keep, COLS]
    writer.write_table(pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False))
    n_written += len(out)
    if i % 5 == 0:
        log(f"normalized {n_written:,} rows")
writer.close()
log(f"normalized.parquet: {n_written:,} rows (empty after normalization: {n_empty_after_norm})")
con.execute(f"CREATE OR REPLACE VIEW norm AS SELECT * FROM read_parquet('{q(NORM)}')")

prod = con.execute("""
  SELECT product_id, count(*) AS n,
    count(*) FILTER (WHERE label='recommended') AS r0,
    count(*) FILTER (WHERE label='not_recommended') AS r1,
    count(*) FILTER (WHERE label='no_idea') AS r2,
    count(*) FILTER (WHERE rating_band='low') AS b0,
    count(*) FILTER (WHERE rating_band='medium') AS b1,
    count(*) FILTER (WHERE rating_band='high') AS b2
  FROM norm GROUP BY 1""").df()
prod["h"] = prod["product_id"].map(
    lambda p: int(hashlib.sha256(f"{SPLIT_SEED}|{p}".encode()).hexdigest()[:16], 16))
prod = prod.sort_values(["n", "h"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

names = list(SPLIT_FRACTIONS)
V = prod[["n", "r0", "r1", "r2", "b0", "b1", "b2"]].to_numpy(dtype=np.float64)
T = np.array([[SPLIT_FRACTIONS[s] * t for t in V.sum(axis=0)] for s in names])
Tsafe = np.where(T > 0, T, 1.0)
cur = np.zeros_like(T)
assign_idx = np.empty(len(V), dtype=np.int8)
for i in range(len(V)):                  
    v = V[i]
    inc = (((cur + v - T) ** 2 - (cur - T) ** 2) / Tsafe ** 2).sum(axis=1)
    s = int(np.argmin(inc)); assign_idx[i] = s; cur[s] += v
assign = pd.DataFrame({"product_id": prod["product_id"], "split": [names[k] for k in assign_idx]})
con.register("assign", assign)
log(f"split done: {assign['split'].value_counts().to_dict()} products")

tr = con.execute("""SELECT n.id, n.label FROM norm n JOIN assign a ON n.product_id = a.product_id
                    WHERE a.split = 'train' AND n.label IS NOT NULL""").df()
tr["h"] = tr["id"].map(stable_hash)
masked_ids, mask_counts = [], {}
for cls, g in tr.groupby("label"):
    k = math.floor(Fraction(str(MASK_RATE)) * len(g))     
    g = g.sort_values(["h", "id"], kind="mergesort")
    masked_ids += g["id"].iloc[:k].tolist()
    mask_counts[cls] = {"labeled_train_rows": int(len(g)), "masked": int(k),
                        "kept_supervised": int(len(g) - k)}
con.register("masked", pd.DataFrame({"id": masked_ids}))
log(f"masked: {mask_counts}")

J = "FROM norm n JOIN assign a ON n.product_id = a.product_id"
USABLE = "(n.label IS NOT NULL OR n.rating_band IS NOT NULL)"
LAB = "n.id, n.body, n.label AS recommendation_status, n.rating_band"
files = {
 "train_supervised.csv": f"SELECT {LAB} {J} LEFT JOIN masked m ON n.id = m.id WHERE a.split='train' AND m.id IS NULL AND {USABLE} ORDER BY n.id",
 "train_pseudo_unlabeled.csv": "SELECT n.id AS id, n.body AS body FROM norm n JOIN masked m ON n.id = m.id ORDER BY n.id",
 "natural_unlabeled.csv": f"SELECT n.id AS id, n.body AS body {J} WHERE a.split='train' AND n.label IS NULL ORDER BY n.id",
 "validation.csv": f"SELECT {LAB} {J} WHERE a.split='validation' AND {USABLE} ORDER BY n.id",
 "test.csv": f"SELECT {LAB} {J} WHERE a.split='test' AND {USABLE} ORDER BY n.id",
}
file_rows = {}
for name, select in files.items():
    path = PROC / name
    con.execute(f"COPY ({select}) TO '{q(path)}' (HEADER, DELIMITER ',')")
    file_rows[name] = con.execute(f"SELECT count(*) FROM read_csv('{q(path)}', all_varchar=true)").fetchone()[0]
    log(f"{name}: {file_rows[name]:,} rows")

assert file_rows["train_pseudo_unlabeled.csv"] == len(masked_ids), "pseudo pool != masked count"
cols = con.execute(f"DESCRIBE SELECT * FROM read_csv('{q(PROC / 'train_pseudo_unlabeled.csv')}', all_varchar=true)").df()["column_name"].tolist()
assert cols == ["id", "body"], cols

stats = con.execute(f"""
  SELECT a.split, count(*) AS rows,
    count(*) FILTER (WHERE {USABLE}) AS usable_rows,
    count(*) FILTER (WHERE n.label='recommended') AS recommended,
    count(*) FILTER (WHERE n.label='not_recommended') AS not_recommended,
    count(*) FILTER (WHERE n.label='no_idea') AS no_idea,
    count(*) FILTER (WHERE n.label IS NULL) AS label_missing,
    count(*) FILTER (WHERE n.rating_band='low') AS band_low,
    count(*) FILTER (WHERE n.rating_band='medium') AS band_medium,
    count(*) FILTER (WHERE n.rating_band='high') AS band_high,
    count(*) FILTER (WHERE n.rating_band IS NULL) AS band_missing
  {J} GROUP BY 1""").df().set_index("split")
tot_rows = int(stats["rows"].sum())

manifest = {
  "seeds": {"SPLIT_SEED": SPLIT_SEED, "MASK_SEED": MASK_SEED, "MASK_RATE": MASK_RATE},
  "split_strategy": "product-grouped; products sorted by (-rows, sha256(SPLIT_SEED|product_id)); "
                    "greedy assignment minimizing normalized deviation of rows, 3 recommendation "
                    "classes and 3 rating bands from 70/15/15 targets",
  "target_fractions": SPLIT_FRACTIONS,
  "achieved_row_fractions": {s: round(float(stats.loc[s, "rows"]) / tot_rows, 4) for s in names},
  "n_products_per_split": assign["split"].value_counts().to_dict(),
  "rating_band": {"thresholds": BAND, "encoding": BAND_CODE, "variant": "B (rate=0 → NULL)"},
  "label_map": LABEL_MAP,
  "normalization": "see src/text_norm.py (ي/ك/ى→ی/ک/ی, digits→ASCII, tags/URLs removed, ZWNJ kept, "
                   "other zero-width removed, letter repeats>=3→2, whitespace collapsed, emojis kept, "
                   "no stopword/negation removal). Output `body` columns are normalized text.",
  "exclusions": {**ex, **rate_diag, "empty_after_normalization": n_empty_after_norm,
                 "duplicate_id_policy": "keep first by (hash(body), product_id)",
                 "note": "reason counts may overlap; rows with invalid rate keep rating_band=NULL"},
  "split_stats": stats.reset_index().to_dict("records"),
  "masking": {"rule": "per recommendation class in train: sort by (stable_hash(id), id), mask floor(MASK_RATE*count)",
              "per_class": mask_counts, "total_masked": len(masked_ids)},
  "files_rows": file_rows,
  "file_notes": {
    "train_supervised.csv": "train rows not masked with >=1 valid target; masked rows removed for BOTH heads",
    "natural_unlabeled.csv": "train rows with missing recommendation label (id, body only)",
    "validation.csv/test.csv": "rows with >=1 valid target; never masked"},
}
for p in (PROC / "split_manifest.json", REP / "split_manifest.json"):
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
print(stats.T)
log("DONE")