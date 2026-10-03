"""Prophet sensitivity: 10 configurations on a stratified sample of 400 municipalities.

Question: is the organizer's base model weak only because of our configuration? Each
configuration is fitted per series and origin (rolling origin, as everywhere) for the origins
of the test window (targets 2024-07..12, h = 1, 3, 6, 12). The table reports every
configuration on the test window. The best one is chosen after the fact, which favours
Prophet; this is on purpose (it is the strictest comparison for our model).

Sample: 400 МО (all 6 categories, 2400 series), 80 from each quintile of the "Все категории"
level in 2023, seed 42.

Output: results/metrics/prophet_grid_test.csv (config, h, MAE on the sample, plus our models
on the same sample) and results/cache/prophet_grid.npz (cubes, sample rows only).

Usage: .venv/bin/python src/models/prophet_grid.py [--n-jobs 8]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, HMAX, ROOT, get_panel, load_config, load_cube  # noqa: E402

CONFIGS = {
    "default": dict(),                                     # Prophet defaults (additive, auto seasonality)
    "ours_m3": dict(yearly=3, mode="multiplicative"),      # our main baseline configuration
    "m5": dict(yearly=5, mode="multiplicative"),
    "add3": dict(yearly=3, mode="additive"),
    "log3": dict(yearly=3, mode="additive", log=True),     # fit on log y
    "m3_cps001": dict(yearly=3, mode="multiplicative", cps=0.01),
    "m3_cps05": dict(yearly=3, mode="multiplicative", cps=0.5),
    "m3_flat": dict(yearly=3, mode="multiplicative", growth="flat"),
    "log3_cps001": dict(yearly=3, mode="additive", log=True, cps=0.01),
    "m3_sps1": dict(yearly=3, mode="multiplicative", sps=1.0),   # shrunk seasonality
}


def sample_rows(meta, Y, n_mo=400, seed=42):
    allc = meta["category"].to_numpy() == "Все категории"
    lvl = pd.Series(Y[allc, :12].mean(1), index=meta.loc[allc, "territory_id"].to_numpy())
    q = pd.qcut(lvl, 5, labels=False)
    rng = np.random.default_rng(seed)
    mos = np.concatenate([rng.choice(q.index[q == k], n_mo // 5, replace=False) for k in range(5)])
    return np.where(meta["territory_id"].isin(mos).to_numpy())[0]


def _fit_chunk(Yc, months, origins, cfg):
    from prophet import Prophet
    warnings.filterwarnings("ignore")
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).disabled = True
    n, T = Yc.shape
    out = np.full((n, T, HMAX), np.nan, dtype=np.float32)
    ds = pd.to_datetime(months)
    fut_all = pd.date_range(ds[0], periods=T + HMAX, freq="MS")
    for o in origins:
        for i in range(n):
            y = np.log(Yc[i, :o + 1]) if cfg.get("log") else Yc[i, :o + 1]
            kw = dict(weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=0)
            if "yearly" in cfg:
                kw["yearly_seasonality"] = cfg["yearly"] if o + 1 >= 12 else False
                kw["seasonality_mode"] = cfg["mode"]
            if "cps" in cfg:
                kw["changepoint_prior_scale"] = cfg["cps"]
            if "sps" in cfg:
                kw["seasonality_prior_scale"] = cfg["sps"]
            if "growth" in cfg:
                kw["growth"] = cfg["growth"]
            m = Prophet(**kw)
            m.fit(pd.DataFrame({"ds": ds[:o + 1], "y": y}))
            p = m.predict(pd.DataFrame({"ds": fut_all[o + 1:o + 1 + HMAX]}))["yhat"].to_numpy()
            out[i, o] = np.exp(p) if cfg.get("log") else np.maximum(p, 0.0)
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    idx = {m: i for i, m in enumerate(months)}
    test = [idx[m] for m in cfg["backtest"]["test_targets"]]
    H = cfg["backtest"]["horizons"]
    origins = sorted({T - h for T in test for h in H})
    rows = sample_rows(meta, Y)
    Ys = Y[rows]
    chunks = np.array_split(np.arange(len(rows)), 48)
    cubes, t0 = {}, time.time()
    for name in a.configs:
        res = Parallel(n_jobs=a.n_jobs)(delayed(_fit_chunk)(Ys[ix], months, origins, CONFIGS[name])
                                        for ix in chunks)
        C = np.full((len(rows), Y.shape[1], HMAX), np.nan, dtype=np.float32)
        for ix, r in zip(chunks, res):
            C[ix] = r
        cubes[name] = C
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(CACHE / "prophet_grid.npz", rows=rows, **cubes)
    others = {m: load_cube(m)[rows] for m in ("prophet", "ensemble", "ensemble_v2") if (CACHE / f"{m}.npz").exists()}
    out = []
    for name, C in {**cubes, **others}.items():
        for h in H:
            y = np.concatenate([Ys[:, T] for T in test])
            p = np.concatenate([C[:, T - h, h - 1] for T in test])
            out.append({"config": name, "h": h, "MAE": float(np.abs(p - y).mean()),
                        "kind": "prophet grid" if name in cubes else "reference"})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "metrics" / "prophet_grid_test.csv", index=False)
    print(df.pivot(index="config", columns="h", values="MAE").round(0).to_string())


if __name__ == "__main__":
    main()
