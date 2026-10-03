"""Data loading for the SberIndex forecasting track.

Builds a wide panel: one row per (territory_id, category) series, one column per month
(2023-01 .. 2024-12). Values are SberIndex estimates of average cashless spending per
resident of the municipality (rubles).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/baseline.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dictionary(cfg: dict) -> pd.DataFrame:
    """One row per territory_id with name, region, type, centre coordinates."""
    d = pd.read_excel(ROOT / cfg["data"]["dictionary"])
    # A territory_id can have several historical versions; keep the latest one.
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    cols = ["territory_id", "municipal_district_name", "municipal_district_type",
            "region_code", "region_name", "municipal_district_center_lat",
            "municipal_district_center_lon"]
    d = d[cols].rename(columns={"municipal_district_name": "name",
                                "municipal_district_type": "mo_type",
                                "municipal_district_center_lat": "lat",
                                "municipal_district_center_lon": "lon"})
    return d


def load_panel(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (meta, Y, months).

    meta : DataFrame with one row per series (territory_id, category + static features)
    Y    : float array, shape (n_series, n_months), spending in rubles
    months : list of 'YYYY-MM' strings
    """
    c = pd.read_parquet(ROOT / cfg["data"]["consumption"])
    wide = c.pivot_table(index=["territory_id", "category"], columns="date",
                         values="value", aggfunc="first")
    months = sorted(wide.columns)
    wide = wide[months]
    keep = wide.notna().sum(axis=1) >= cfg["data"]["min_months"]
    wide = wide[keep & (wide > 0).all(axis=1)]

    meta = wide.index.to_frame(index=False)
    ma = pd.read_parquet(ROOT / cfg["data"]["market_access"])
    meta = meta.merge(ma, on="territory_id", how="left")
    meta = meta.merge(load_dictionary(cfg), on="territory_id", how="left")
    return meta, wide.to_numpy(dtype=float), months
