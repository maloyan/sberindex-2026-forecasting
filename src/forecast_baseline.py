"""Forecasting baseline: rolling-origin backtest + 2025 forecast.

Models
  naive          y_hat(T) = y(o)
  snaive         y_hat(T) = y(T-12)
  snaive_growth  y(T-12) scaled by the series' own year-over-year growth at the origin
  prophet        per-series Prophet (on a fixed sample of territories, it is slow)
  lgbm           one global LightGBM, direct multi-horizon, target = log(y_T / y_o)

Backtest
  For each horizon h in {1,3,6,12} and each target month T in 2024-07..2024-12,
  the origin is o = T - h. A model sees only months <= o. Metrics are pooled over
  all series and all target months: MAE (rubles, mandatory), R2, WAPE, MdAPE.

Run:  .venv/bin/python src/forecast_baseline.py [--no-prophet]
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import ROOT, load_config, load_panel

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    ss_res = ((y - p) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"MAE": err.mean(), "R2": 1 - ss_res / ss_tot,
            "WAPE": err.sum() / np.abs(y).sum(), "MdAPE": np.median(err / np.abs(y)),
            "n": len(y)}


# ----------------------------------------------------------------------------- simple models
def pred_naive(Y, o, h):
    return Y[:, o]


def pred_snaive(Y, o, h):
    T = o + h
    return Y[:, T - 12] if T - 12 >= 0 else Y[:, o]


def pred_snaive_growth(Y, o, h, k=3):
    """y(T-12) * exp(mean YoY log growth over the last k months that have a year-ago value)."""
    T = o + h
    base = pred_snaive(Y, o, h)
    kk = min(k, o - 11)            # months j <= o with j-12 >= 0
    if kk <= 0 or T - 12 < 0:
        return base
    L = np.log(Y)
    g = (L[:, o - kk + 1:o + 1] - L[:, o - kk + 1 - 12:o + 1 - 12]).mean(axis=1)
    return base * np.exp(g)


# ----------------------------------------------------------------------------- prophet
def _prophet_one(y_hist: np.ndarray, months_hist: list[str], hmax: int, pcfg: dict):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").disabled = True
    df = pd.DataFrame({"ds": pd.to_datetime(months_hist), "y": y_hist})
    yearly = pcfg["yearly_seasonality"] if len(y_hist) >= 12 else False
    m = Prophet(yearly_seasonality=yearly, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=pcfg["changepoint_prior_scale"],
                seasonality_mode=pcfg["seasonality_mode"], uncertainty_samples=0)
    m.fit(df)
    fut = m.make_future_dataframe(periods=hmax, freq="MS", include_history=False)
    return np.maximum(m.predict(fut)["yhat"].to_numpy(), 0.0)


def prophet_forecasts(Y, months, origins, hmax, pcfg, n_jobs=-1):
    """Return dict origin -> array (n_series, hmax)."""
    from joblib import Parallel, delayed
    out = {}
    for o in origins:
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(
            delayed(_prophet_one)(Y[i, :o + 1], months[:o + 1], hmax, pcfg)
            for i in range(Y.shape[0]))
        out[o] = np.vstack(res)
        print(f"    prophet origin {months[o]}: {time.time() - t0:.1f}s", flush=True)
    return out


# ----------------------------------------------------------------------------- global lgbm
CAT_FEATS = ["category", "region_code", "mo_type"]


def build_rows(Y, meta, months, pairs):
    """Feature rows for (origin o, horizon h) pairs. Features are taken at origin o only."""
    L = np.log(Y)
    n = Y.shape[0]
    mon = np.array([int(m[5:7]) for m in months])
    # cross-sectional (national, per category) monthly log change, known at time t
    dl = np.full_like(L, np.nan)
    dl[:, 1:] = L[:, 1:] - L[:, :-1]
    cat_codes = meta["category"].astype("category").cat.codes.to_numpy()
    nat = np.full_like(L, np.nan)
    for c in np.unique(cat_codes):
        msk = cat_codes == c
        nat[msk] = np.nanmedian(dl[msk], axis=0)

    frames = []
    for o, h in pairs:
        T = o + h
        f = {"h": np.full(n, h), "month_T": np.full(n, mon[T] if T < len(mon) else (mon[o] + h - 1) % 12 + 1),
             "month_o": np.full(n, mon[o]), "log_level": L[:, o]}
        for k in (1, 2, 3, 6, 9, 11, 12):
            f[f"lag{k}"] = L[:, o - k] - L[:, o] if o - k >= 0 else np.full(n, np.nan)
        # seasonal anchor: same calendar month one year before the target, relative to origin
        f["seas"] = L[:, T - 12] - L[:, o] if 0 <= T - 12 <= o else np.full(n, np.nan)
        f["seas_m1"] = L[:, T - 13] - L[:, o] if 0 <= T - 13 <= o else np.full(n, np.nan)
        f["yoy"] = L[:, o] - L[:, o - 12] if o >= 12 else np.full(n, np.nan)
        f["mean3"] = L[:, max(0, o - 2):o + 1].mean(axis=1) - L[:, o]
        f["nat_d1"] = nat[:, o]
        f["nat_d3"] = np.nansum(nat[:, max(1, o - 2):o + 1], axis=1)
        df = pd.DataFrame(f)
        df["o"] = o
        df["T"] = T
        df["sid"] = np.arange(n)
        df["y_o"] = Y[:, o]
        df["target"] = L[:, T] - L[:, o] if T < Y.shape[1] else np.nan
        frames.append(df)
    X = pd.concat(frames, ignore_index=True)
    static = meta[["category", "region_code", "mo_type", "market_access", "lat", "lon"]].copy()
    for c in CAT_FEATS:
        static[c] = static[c].astype("category")
    X = X.join(static, on="sid")
    return X


FEATS = None


def lgbm_fit_predict(Y, meta, months, train_cutoff, pred_pairs, params, hmax=12, seed=42):
    """Train on all (o,h) with o>=3 and o+h <= train_cutoff, predict pred_pairs."""
    global FEATS
    tr_pairs = [(o, h) for o in range(3, train_cutoff) for h in range(1, hmax + 1)
                if o + h <= train_cutoff]
    Xtr = build_rows(Y, meta, months, tr_pairs)
    Xte = build_rows(Y, meta, months, pred_pairs)
    FEATS = [c for c in Xtr.columns if c not in ("o", "T", "sid", "y_o", "target")]
    p = dict(params)
    nround = p.pop("num_boost_round")
    p["seed"] = seed
    # weight by level: L1 on log-ratio * level ~ absolute error in rubles
    dtr = lgb.Dataset(Xtr[FEATS], Xtr["target"], weight=Xtr["y_o"] / Xtr["y_o"].mean(),
                      categorical_feature=CAT_FEATS)
    model = lgb.train(p, dtr, num_boost_round=nround)
    Xte["pred"] = Xte["y_o"] * np.exp(model.predict(Xte[FEATS]))
    return Xte, model


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--no-prophet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bt = cfg["backtest"]
    rng = np.random.default_rng(bt["seed"])

    meta, Y, months = load_panel(cfg)
    print(f"series: {Y.shape[0]}  months: {months[0]}..{months[-1]}", flush=True)
    idx = {m: i for i, m in enumerate(months)}
    targets = [idx[m] for m in bt["target_months"]]
    H = bt["horizons"]

    terr = meta["territory_id"].unique()
    sample_terr = rng.choice(terr, size=min(bt["prophet_sample_territories"], len(terr)),
                             replace=False)
    in_sample = meta["territory_id"].isin(sample_terr).to_numpy()

    rows = []  # long table: sid, T, h, model, y, pred

    def add(model, sid, T, h, y, p):
        rows.append(pd.DataFrame({"sid": sid, "T": T, "h": h, "model": model, "y": y, "pred": p}))

    sid_all = np.arange(Y.shape[0])
    for h in H:
        for T in targets:
            o = T - h
            for name, fn in [("naive", pred_naive), ("snaive", pred_snaive),
                             ("snaive_growth", pred_snaive_growth)]:
                add(name, sid_all, T, h, Y[:, T], fn(Y, o, h))

    # LightGBM: one training per distinct origin (train targets <= origin)
    origins = sorted({T - h for h in H for T in targets})
    t0 = time.time()
    for o in origins:
        pairs = [(o, h) for h in H if o + h in targets]
        Xte, _ = lgbm_fit_predict(Y, meta, months, o, pairs, cfg["models"]["lgbm"], seed=bt["seed"])
        for (oo, h), g in Xte.groupby(["o", "h"]):
            add("lgbm", g["sid"].to_numpy(), oo + h, h, Y[g["sid"], oo + h], g["pred"].to_numpy())
    print(f"lgbm backtest: {time.time() - t0:.0f}s", flush=True)

    if not args.no_prophet:
        Ys = Y[in_sample]
        sids = sid_all[in_sample]
        print(f"prophet on {Ys.shape[0]} series x {len(origins)} origins", flush=True)
        pf = prophet_forecasts(Ys, months, origins, max(H), cfg["models"]["prophet"])
        for h in H:
            for T in targets:
                add("prophet", sids, T, h, Y[sids, T], pf[T - h][:, h - 1])

    res = pd.concat(rows, ignore_index=True)
    res["category"] = meta["category"].to_numpy()[res["sid"]]
    res["in_sample"] = in_sample[res["sid"]]
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    res.to_parquet(out_dir / "backtest_predictions.parquet")

    def table(df):
        t = (df.groupby(["h", "model"]).apply(lambda g: pd.Series(metrics(g["y"], g["pred"])))
             .reset_index())
        return t

    full = table(res[res.model != "prophet"])
    full.to_csv(out_dir / "metrics_full.csv", index=False)
    samp = table(res[res.in_sample])
    samp.to_csv(out_dir / "metrics_prophet_sample.csv", index=False)
    bycat = (res[res.model != "prophet"].groupby(["category", "h", "model"])
             .apply(lambda g: pd.Series(metrics(g["y"], g["pred"]))).reset_index())
    bycat.to_csv(out_dir / "metrics_by_category.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== all series (no prophet) ===")
    print(full.pivot(index="model", columns="h", values="MAE").round(1))
    print(full.pivot(index="model", columns="h", values="R2").round(4))
    print("\n=== prophet sample (common series) ===")
    print(samp.pivot(index="model", columns="h", values="MAE").round(1))
    print(samp.pivot(index="model", columns="h", values="WAPE").round(4))

    # ---- final 2025 forecast from origin 2024-12 with the global model
    fc = cfg["forecast"]
    o = idx[fc["origin"]]
    pairs = [(o, h) for h in range(1, fc["horizon"] + 1)]
    Xte, model = lgbm_fit_predict(Y, meta, months, o, pairs, cfg["models"]["lgbm"], seed=bt["seed"])
    future = pd.period_range(pd.Period(fc["origin"], "M") + 1, periods=fc["horizon"], freq="M")
    out = pd.DataFrame({
        "territory_id": meta["territory_id"].to_numpy()[Xte["sid"]],
        "category": meta["category"].to_numpy()[Xte["sid"]],
        "date": [str(future[h - 1]) for h in Xte["h"]],
        "horizon": Xte["h"].to_numpy(),
        "forecast": Xte["pred"].round(1).to_numpy(),
    })
    out.to_csv(ROOT / fc["out"], index=False)
    imp = pd.Series(model.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    imp.to_csv(out_dir / "lgbm_feature_importance.csv")
    print(f"\nwrote {fc['out']}: {len(out)} rows; top features: {list(imp.index[:6])}")


if __name__ == "__main__":
    main()
