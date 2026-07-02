"""Merge the surviving new factors (artifacts/new_factors.parquet) into an EXTENDED
feature set. Writes only NEW files -- artifacts/features.parquet / feature_list.json
are never touched.

Output: artifacts/features_nf.parquet, artifacts/feature_list_nf.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/root/autodl-tmp/eg_model")

base = pd.read_parquet(ROOT / "artifacts" / "features.parquet")
base_cols = json.loads((ROOT / "artifacts" / "feature_list.json").read_text())
nf = pd.read_parquet(ROOT / "artifacts" / "new_factors.parquet")
nf_cols = json.loads((ROOT / "artifacts" / "new_factor_list.json").read_text())

print(f"[merge] base features {base.shape}, new factors {nf.shape} ({len(nf_cols)} cols)", flush=True)
merged = base.merge(nf, on=["day", "instrument_id"], how="left")
for c in nf_cols:
    merged[c] = merged[c].fillna(0.0).astype("float32")

out_cols = base_cols + nf_cols
merged = merged.sort_values(["day", "instrument_id"]).reset_index(drop=True)
merged.to_parquet(ROOT / "artifacts" / "features_nf.parquet", index=False)
(ROOT / "artifacts" / "feature_list_nf.json").write_text(json.dumps(out_cols, indent=0))
print(f"[merge] wrote features_nf.parquet {merged.shape}, {len(out_cols)} total feature cols "
      f"({len(base_cols)} original + {len(nf_cols)} new)", flush=True)
