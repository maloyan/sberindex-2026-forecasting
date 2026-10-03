"""Out-of-sample check of the 2025 forecast against real 2025 national data (SberIndex portal).

The 2025 forecast (results/forecast_2025.csv) is made at origin 2024-12 and uses portal data
up to 2024-12 only. The portal now has 2025 actuals, so the national growth of our forecast
can be checked. The panel is per resident and municipal; the portal is a Russia total. So we
compare YoY growth, not levels:

  ours     median over МО of log(forecast 2025-m / actual 2024-m), per category
  portal   log(portal 2025-m / portal 2024-m) for the portal type of the category
  gap 2024 the same two YoY numbers for 2024 (panel actual vs portal), to show how far the
           two sources differ even with no forecast error

Output: results/metrics/check_2025_portal.csv
Usage: .venv/bin/python src/check_2025.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, get_panel, load_config  # noqa: E402
from exog.portal import PORTAL_FOR, load_portal  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", nargs="*", default=None,
                    help="final cubes (origin 2024-12) to check instead of forecast_2025.csv")
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    P = np.log(load_portal())
    cats = meta["category"].to_numpy()
    if a.cubes:
        from common import load_cube
        out = []
        for name in a.cubes:
            fc = cube_frame(load_cube(name), meta, months)
            out.append(check(fc, meta, Y, months, P, cats).assign(source=name))
        df = pd.concat(out)
        df.to_csv(ROOT / "results" / "metrics" / "check_2025_portal_cubes.csv", index=False)
        print((df.groupby(["source", "category"])[["yoy_forecast", "yoy_portal_2025", "err_2025"]].mean() * 100)
              .round(1).to_string())
        print((df.groupby("source").err_2025.apply(lambda e: e.abs().mean()) * 100).round(2).to_string())
        return
    fc = pd.read_csv(ROOT / cfg["forecast"]["out"])
    df = check(fc, meta, Y, months, P, cats)
    df.to_csv(ROOT / "results" / "metrics" / "check_2025_portal.csv", index=False)
    s = df.groupby("category")[["yoy_forecast", "yoy_portal_2025", "err_2025", "gap_2024"]].mean() * 100
    print(s.round(1).to_string())
    print("mean |err_2025| pp:", (df.err_2025.abs().mean() * 100).round(2),
          " mean |gap_2024| pp:", (df.gap_2024.abs().mean() * 100).round(2))


def cube_frame(C, meta, months):
    o = len(months) - 1
    base = pd.Period(months[-1], "M")
    return pd.concat([pd.DataFrame({"territory_id": meta["territory_id"].to_numpy(),
                                    "category": meta["category"].to_numpy(),
                                    "date": str(base + h), "forecast": C[:, o, h - 1]})
                      for h in range(1, 13)])


def check(fc, meta, Y, months, P, cats):
    last = pd.DataFrame({"territory_id": meta["territory_id"].to_numpy(), "category": cats})
    rows = []
    for m in range(1, 13):
        d25, d24, d23 = f"2025-{m:02d}", f"2024-{m:02d}", f"2023-{m:02d}"
        if d25 not in P.index:
            continue
        f = fc[fc.date == d25].merge(last.assign(y24=Y[:, months.index(d24)]), on=["territory_id", "category"])
        panel24 = pd.Series(np.log(Y[:, months.index(d24)] / Y[:, months.index(d23)])).groupby(cats).median()
        for c, g in f.groupby("category"):
            col = PORTAL_FOR[c]
            rows.append({"month": d25, "category": c, "portal_type": col,
                         "yoy_forecast": float(np.median(np.log(g.forecast / g.y24))),
                         "yoy_portal_2025": float(P.loc[d25, col] - P.loc[d24, col]),
                         "yoy_panel_2024": float(panel24[c]),
                         "yoy_portal_2024": float(P.loc[d24, col] - P.loc[d23, col])})
    df = pd.DataFrame(rows)
    df["err_2025"] = df.yoy_forecast - df.yoy_portal_2025
    df["gap_2024"] = df.yoy_panel_2024 - df.yoy_portal_2024
    return df


if __name__ == "__main__":
    main()
