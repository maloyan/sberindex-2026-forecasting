"""Rosstat monthly national turnover (open data, rosstat.gov.ru, free reuse with a reference).

Sources (downloaded to data/raw/rosstat/ by `make data`):
  Oborot_2025.xls          sheet 2: retail turnover by month, RF, current prices, 2000-2025
                           (total, food, non-food)
  Obschepit_2000-2025.xls  sheet 1: food service turnover by month, RF, 2020-2025

Output: data/external/rosstat_monthly.csv  (date, retail_total, retail_food, retail_nonfood,
food_service) and data/external/seasonal_prior.csv (month 1..12 x series: log seasonal factor
estimated ONLY on years before 2023, so it carries no information from the SberIndex period).

Usage: .venv/bin/python src/exog/rosstat.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT  # noqa: E402

MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
          "сентябрь", "октябрь", "ноябрь", "декабрь"]
RAW = ROOT / "data" / "raw" / "rosstat"
EXT = ROOT / "data" / "external"


def _clean(s) -> str:
    return re.sub(r"[^а-яa-z0-9\-]", "", str(s).lower())


def parse_long_sheet(path, sheet, cols: dict[int, str]) -> pd.DataFrame:
    """Sheet with year header rows followed by month rows (retail, sheet 2)."""
    x = pd.read_excel(path, sheet_name=sheet, header=None)
    year, rows = None, []
    for _, r in x.iterrows():
        c0 = _clean(r[0])
        m = re.match(r"^(\d{4})", c0)
        if m and pd.isna(r[1]):
            year = int(m.group(1)); continue
        if year and c0 in MONTHS:
            rows.append({"date": f"{year}-{MONTHS.index(c0) + 1:02d}",
                         **{name: float(r[j]) for j, name in cols.items()}})
    return pd.DataFrame(rows)


def parse_wide_sheet(path, sheet, name) -> pd.DataFrame:
    """Sheet with years in columns and months in rows (food service, sheet 1)."""
    x = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = None
    for i in range(len(x)):
        ys = [re.match(r"^(\d{4})", _clean(v)) for v in x.iloc[i, 1:]]
        if sum(bool(y) for y in ys) >= 3:
            hdr = {j + 1: int(y.group(1)) for j, y in enumerate(ys) if y}
            start = i + 1
            break
    rows = []
    for i in range(start, len(x)):
        c0 = _clean(x.iloc[i, 0])
        if c0 in MONTHS:
            for j, yr in hdr.items():
                v = pd.to_numeric(x.iloc[i, j], errors="coerce")
                if pd.notna(v):
                    rows.append({"date": f"{yr}-{MONTHS.index(c0) + 1:02d}", name: float(v)})
    return pd.DataFrame(rows)


def seasonal_profile(s: pd.Series, years: list[int]) -> pd.Series:
    """Log seasonal factor by calendar month: log(x) minus a centred 2x12 moving average."""
    L = np.log(s)
    trend = L.rolling(12, center=True).mean().rolling(2).mean().shift(-1)
    dev = (L - trend).dropna()
    dev = dev[[int(d[:4]) in years for d in dev.index]]
    prof = dev.groupby([int(d[5:7]) for d in dev.index]).mean()
    return prof - prof.mean()


def main():
    EXT.mkdir(parents=True, exist_ok=True)
    r = parse_long_sheet(RAW / "Oborot_2025.xls", "2",
                         {1: "retail_total", 2: "retail_food", 3: "retail_nonfood"})
    f = parse_wide_sheet(RAW / "Obschepit_2000-2025.xls", "1", "food_service")
    df = r.merge(f, on="date", how="outer").sort_values("date").drop_duplicates("date")
    df.to_csv(EXT / "rosstat_monthly.csv", index=False)
    s = df.set_index("date")
    prior_years = {"retail_total": [2015, 2016, 2017, 2018, 2019, 2021, 2022],
                   "retail_food": [2015, 2016, 2017, 2018, 2019, 2021, 2022],
                   "retail_nonfood": [2015, 2016, 2017, 2018, 2019, 2021, 2022],
                   "food_service": [2021, 2022]}
    prof = pd.DataFrame({k: seasonal_profile(s[k].dropna(), v) for k, v in prior_years.items()})
    prof.index.name = "month"
    prof.to_csv(EXT / "seasonal_prior.csv")
    print(df.tail(3).to_string()); print(prof.round(3).to_string())


if __name__ == "__main__":
    main()
