"""SberIndex portal national consumer spending (open data, sberindex.ru), monthly 2018-12 onward.

Source: https://sberindex.ru/api/dataset/v1/download/consumer-spending/parquet
(downloaded by `make data` to data/raw/portal/consumer_spending.parquet). Types: Всего,
Продовольственные товары, Непродовольственные товары, Общественное питание, Услуги; billion
rubles, Russia total.

Vintage. The file is the portal state on the download date (first download 2026-10-03). The
backtest uses, at origin o, only portal months <= o (the national aggregate of a month is
published before the municipal data of that month). Past values may have been revised since
2024; we cannot get the 2024 vintage, so this is a stated limitation.

Output: data/external/portal_national.csv (date x type, billion rubles).
Usage: .venv/bin/python src/exog/portal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

RAW = ROOT / "data" / "raw" / "portal" / "consumer_spending.parquet"
OUT = ROOT / "data" / "external" / "portal_national.csv"

# category of the contest panel -> portal type (domain mapping, fixed before validation)
PORTAL_FOR = {"Все категории": "Всего", "Продовольствие": "Продовольственные товары",
              "Общественное питание": "Общественное питание",
              "Маркетплейсы": "Непродовольственные товары", "Здоровье": "Всего", "Транспорт": "Всего"}


def load_portal() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_csv(OUT, index_col=0)
    return build()


def build() -> pd.DataFrame:
    d = pd.read_parquet(RAW)
    w = d.pivot(index="period", columns="type", values="value")
    w.index = w.index.str[:7]
    w.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(OUT)
    return w


if __name__ == "__main__":
    print(build().tail(3).to_string())
