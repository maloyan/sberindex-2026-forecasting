"""Prophet (organizer's base model): one model per series and origin, full panel.

Monthly data, yearly seasonality as a Fourier series of order 3 (only 1-2 years of
history), multiplicative mode, no weekly/daily terms. Origins with fewer than
`min_history` months use the naive forecast.

Usage: .venv/bin/python src/models/prophet_model.py [--n-jobs 8]
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
from common import HMAX, empty_cube, get_panel, load_config, save_cube  # noqa: E402


def _fit_chunk(Yc: np.ndarray, months: list[str], pcfg: dict) -> np.ndarray:
    from prophet import Prophet
    warnings.filterwarnings("ignore")
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).disabled = True
    n, T = Yc.shape
    out = np.full((n, T, HMAX), np.nan, dtype=np.float32)
    ds = pd.to_datetime(months)
    fut_all = pd.date_range(ds[0], periods=T + HMAX, freq="MS")
    for o in range(T):
        if o + 1 < pcfg["min_history"]:
            out[:, o, :] = Yc[:, o:o + 1]
            continue
        for i in range(n):
            df = pd.DataFrame({"ds": ds[:o + 1], "y": Yc[i, :o + 1]})
            m = Prophet(yearly_seasonality=pcfg["yearly_seasonality"] if o + 1 >= 12 else False,
                        weekly_seasonality=False, daily_seasonality=False,
                        changepoint_prior_scale=pcfg["changepoint_prior_scale"],
                        seasonality_mode=pcfg["seasonality_mode"], uncertainty_samples=0)
            m.fit(df)
            fut = pd.DataFrame({"ds": fut_all[o + 1:o + 1 + HMAX]})
            out[i, o] = np.maximum(m.predict(fut)["yhat"].to_numpy(), 0.0)
    return out


def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=192)
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    t0 = time.time()
    idx = np.array_split(np.arange(Y.shape[0]), a.chunks)
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(_fit_chunk)(Y[ix], months, cfg["models"]["prophet"]) for ix in idx)
    cube = empty_cube(*Y.shape)
    for ix, r in zip(idx, res):
        cube[ix] = r
    save_cube("prophet", cube, seconds=time.time() - t0)
    print(f"prophet done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
