"""Final forecast for 2025-01..2025-12 from origin 2024-12 with the final ensemble.

The ensemble is `forecast.ensemble` in configs/config.yaml: the headline v1 ("ensemble",
src/ensemble.py; weights re-selected on validation + test -> ensemble_final; the models are the
same as in the backtest). "ensemble_v2" (src/stack.py) is a post-hoc experiment. The 80% interval comes from
the empirical quantiles (10%, 90%) of the log errors of the honest ensemble on the test
window, per horizon and category (a split-conformal style interval).

Output: results/forecast_2025.csv
  territory_id, municipality, region, category, date, horizon, forecast, lo80, hi80
plus results/forecast_2025_summary.csv (median growth per category and month).

Usage: .venv/bin/python src/forecast_final.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HMAX, ROOT, gather, get_panel, load_config, load_cube, window_pairs  # noqa: E402
from evaluate import windows  # noqa: E402


def build(cfg, meta, Y, months, C, honest) -> pd.DataFrame:
    """Forecast table from the final cube C (origin = forecast.origin) and the honest cube
    (its test-window log errors give the 80% interval per horizon and category)."""
    fc = cfg["forecast"]
    o = months.index(fc["origin"])
    W = windows(cfg, months)
    cats = meta["category"].to_numpy()
    lo = np.zeros((len(meta), HMAX)); hi = np.zeros((len(meta), HMAX))
    for h in range(1, HMAX + 1):
        y, p, _, _, sid = gather(honest, Y, window_pairs(W["test"], [h]))
        e = np.log(y / p)
        for c in np.unique(cats):
            m = cats[sid] == c
            ql, qh = np.quantile(e[m], [0.1, 0.9])
            lo[cats == c, h - 1], hi[cats == c, h - 1] = ql, qh
    future = pd.period_range(pd.Period(fc["origin"], "M") + 1, periods=fc["horizon"], freq="M")
    rows = []
    for h in range(1, fc["horizon"] + 1):
        p = C[:, o, h - 1].astype(float)
        rows.append(pd.DataFrame({
            "territory_id": meta["territory_id"].to_numpy(), "municipality": meta["name"].to_numpy(),
            "region": meta["region_name"].to_numpy(), "category": cats,
            "date": str(future[h - 1]), "horizon": h, "forecast": p.round(1),
            "lo80": (p * np.exp(lo[:, h - 1])).round(1), "hi80": (p * np.exp(hi[:, h - 1])).round(1)}))
    return pd.concat(rows, ignore_index=True)


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    fc = cfg["forecast"]
    o = months.index(fc["origin"])
    ens = fc.get("ensemble", "ensemble")
    cats = meta["category"].to_numpy()
    out = build(cfg, meta, Y, months, load_cube(f"{ens}_final"), load_cube(ens))
    out.to_csv(ROOT / fc["out"], index=False)
    last = pd.DataFrame({"category": cats, "y2024_12": Y[:, o]})
    g = out.merge(last.assign(territory_id=meta["territory_id"].to_numpy()), on=["territory_id", "category"])
    g["ratio_to_2024_12"] = g["forecast"] / g["y2024_12"]
    summ = g.groupby(["category", "date"]).agg(median_forecast=("forecast", "median"),
                                               median_ratio_to_dec2024=("ratio_to_2024_12", "median")).reset_index()
    summ.to_csv(ROOT / "results" / "forecast_2025_summary.csv", index=False)
    print(f"wrote {fc['out']}: {len(out)} rows")
    print(summ.pivot(index="date", columns="category", values="median_ratio_to_dec2024").round(3).to_string())


if __name__ == "__main__":
    main()
