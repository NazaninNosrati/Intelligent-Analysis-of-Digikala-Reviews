from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"; OUT = PROC / "subset"
ids = pd.read_csv(OUT / "pseudo_sub.csv", dtype=str, usecols=["id"])
con = duckdb.connect(); con.register("ids", ids)
ref = con.execute(f"""SELECT n.id, n.label AS recommendation_status
    FROM read_parquet('{(PROC / 'normalized.parquet').as_posix()}') n JOIN ids USING (id)""").df()
ref.to_csv(OUT / "pseudo_reference_sub.csv", index=False)
print(len(ids), len(ref))     