#!/usr/bin/env python3
"""Share of the retrieval pool's distinct passages on which the GloBI interaction-term scanner
(src/data/interaction_taxonomy.py, 591 terms) fires; Paper A, Setting. -> results/paperA_v2/pool_scan.json"""
import json, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.interaction_taxonomy import _build_globi_scanner  # noqa: E402
scanner = _build_globi_scanner()
p = pd.read_parquet(REPO / "data/training/distill/med25_pool.parquet")
u = p.passage.astype(str).drop_duplicates()
hits = sum(any(pat.search(s.lower()) for _, pat in scanner) for s in u)
out = {"rows": int(len(p)), "distinct_passages": int(len(u)), "distinct_triplet_keys": int(p.triplet_key.nunique()),
       "globi_terms": len(scanner), "passages_with_globi_term": int(hits), "share": hits / len(u)}
(REPO / "results/paperA_v2/pool_scan.json").write_text(json.dumps(out, indent=2)); print(out)
