"""News features per series and month for the forecasting models and the change-point prior.

All features at month t use GDELT documents published up to the end of month t, so a
model at origin o may use them (no look-ahead).

  mo_docs_log      log(1 + documents about the МО in month t)
  mo_docs_surp     mo_docs_log minus its mean over the previous 6 months
  mo_disaster_log  log(1 + documents with a disaster, flood or fire theme)
  mo_econneg_log   log(1 + documents with unemployment, bankruptcy, debt, strike themes)
  mo_security_log  log(1 + documents with military, terror, drone themes)
  mo_tone          mean GDELT tone of the documents (NaN if none)
  reg_*            the same for the region (all places in the region, incl. unmatched)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ROOT

EXT = ROOT / "data" / "external"


def _grid(df, key, months, col):
    p = df.pivot_table(index=key, columns="month", values=col, aggfunc="sum")
    return p.reindex(columns=months).fillna(0.0)


def _surprise(x):
    """x: (n, T) log counts -> x minus the mean of the previous 6 months (0 if no history)."""
    prev = pd.DataFrame(x).T.rolling(6, min_periods=1).mean().shift(1).T.to_numpy()
    return np.nan_to_num(x - prev, nan=0.0)


def news_array(cfg, meta, months):
    mo = pd.read_parquet(EXT / "news_monthly_mo.parquet")
    rg = pd.read_parquet(EXT / "news_monthly_region.parquet")
    feats, names = [], []

    def add(level, df, key, ids):
        df = df.copy()
        df["dis"] = df["g_disaster"] + df["g_flood"] + df["g_fire"]
        docs = _grid(df, key, months, "docs").reindex(ids).fillna(0).to_numpy()
        tone = _grid(df, key, months, "tone_sum").reindex(ids).fillna(0).to_numpy()
        logd = np.log1p(docs)
        out = {f"{level}_docs_log": logd, f"{level}_docs_surp": _surprise(logd),
               f"{level}_disaster_log": np.log1p(_grid(df, key, months, "dis").reindex(ids).fillna(0).to_numpy()),
               f"{level}_econneg_log": np.log1p(_grid(df, key, months, "g_economy_negative").reindex(ids).fillna(0).to_numpy()),
               f"{level}_security_log": np.log1p(_grid(df, key, months, "g_security").reindex(ids).fillna(0).to_numpy()),
               f"{level}_tone": np.where(docs > 0, tone / np.maximum(docs, 1), np.nan)}
        for k, v in out.items():
            feats.append(v); names.append(k)

    add("mo", mo, "territory_id", meta["territory_id"].to_numpy())
    add("reg", rg, "region_code", meta["region_code"].to_numpy())
    return np.stack(feats, axis=2).astype(np.float32), names
