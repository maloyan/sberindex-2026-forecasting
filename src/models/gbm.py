"""Global gradient boosting on the local component (LightGBM and CatBoost), direct multi-horizon.

Target:   d[i,T] - d[i,o]   (change of the local component, see src/models/decomp.py)
Forecast: y_hat = exp(n_hat[c,T|o] + d[i,o] + target_hat)

One model per origin o is trained on all pairs (o', h') with o' + h' <= o (strictly
causal: no target after the origin enters training). Rows are weighted by y[i,o'] so the
L1 loss on log changes approximates the MAE in rubles.

Feature groups (switch them in configs/config.yaml -> models.gbm_v2 or with --groups):
  base      h, calendar month of T and o, category, region, MO type, market access,
            lat/lon, log level, local lags d[o]-d[o-k], local seasonal lag d[T-12]-d[o],
            local volatility, regional mean of d and its change, national growth
  calendar  working days and calendar days of T vs o and vs T-12 (production calendar)
  fm        foundation-model forecasts of the local component (log ratio to d[o]) and
            their 80% interval width -> FM knowledge enters the GBM as features
  news      GDELT news features of the municipality and its region at origin o
  cpd       online change-point features at origin o (break flag, months since break)
  xseries   cross-series features: local change of the same МО in "Все категории" and in
            all its categories, regional 1-month change, rank of d inside region x category

v3 options (src/models/gbm.py --help): national v2 offset, recency weights, target
clipping, full target (log y_T - log y_o, the model also learns the national growth), seeds.

Usage:
  .venv/bin/python src/models/gbm.py --name gbm_lgb --algo lgbm --groups base calendar fm news
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, HMAX, ROOT, empty_cube, get_panel, load_config, save_cube  # noqa: E402
from models.decomp import national_paths  # noqa: E402

CAT = ["category", "region_code", "mo_type"]


class Features:
    def __init__(self, cfg, meta, Y, months, groups, fm_names=None, national="v1"):
        self.meta, self.Y, self.months, self.groups = meta, Y, months, set(groups)
        self.L = np.log(Y)
        cats = meta["category"].to_numpy()
        self.nat = national_paths(self.L, cats)
        self.d = self.L - self.nat
        self.n, self.T = Y.shape
        nat_file = {"v1": "feat_national_log.npz", "v2": "feat_national_log_v2.npz",
                    "v3": "feat_national_log_v3.npz"}[national]
        with np.load(CACHE / nat_file) as z:
            self.N = z["p50"]
        self.mon = np.array([int(m[5:7]) for m in months])
        # regional mean of the local component, per (region, category)
        key = meta["region_code"].astype(str) + "|" + meta["category"].astype(str)
        self.reg = pd.DataFrame(self.d).groupby(key.to_numpy()).transform("mean").to_numpy()
        static = meta[["category", "region_code", "mo_type", "market_access", "lat", "lon"]].copy()
        for c in CAT:
            static[c] = static[c].astype(str).astype("category")
        self.static = static.reset_index(drop=True)
        if "calendar" in self.groups:
            from exog.calendar_ru import calendar_table
            cal = calendar_table().set_index("date")
            self.cal = cal
        if "xseries" in self.groups:
            terr = meta["territory_id"].to_numpy()
            allc = meta["category"].to_numpy() == "Все категории"
            pos = pd.Series(np.where(allc)[0], index=terr[allc])
            self.d_all = self.d[pos.loc[terr].to_numpy()]               # d of the МО total
            self.d_mo = pd.DataFrame(self.d).groupby(terr).transform("mean").to_numpy()
            self.reg_rank = pd.DataFrame(self.d).groupby(key.to_numpy()).rank(pct=True).to_numpy()
        if "fm" in self.groups:
            self.fm = {}
            for nm in fm_names or []:
                p = CACHE / f"fmlocal_{nm}.npz"
                if p.exists():
                    with np.load(p) as z:
                        self.fm[nm] = {k: np.log(np.maximum(z[k], 1e-6)) for k in ("p10", "p50", "p90")}
        if "news" in self.groups:
            from news.features import news_array
            self.news, self.news_names = news_array(cfg, meta, months)
        if "cpd" in self.groups:
            from cpd.features import cpd_array
            self.cpd, self.cpd_names = cpd_array(cfg, meta, Y, months)

    def month_at(self, t):
        return (pd.Period(self.months[0], "M") + t)

    def block(self, o: int, h: int) -> pd.DataFrame:
        if not hasattr(self, "_cache"):
            self._cache = {}
        if (o, h) not in self._cache:
            self._cache[(o, h)] = self._block(o, h)
        return self._cache[(o, h)]

    def _block(self, o: int, h: int) -> pd.DataFrame:
        n, d, T = self.n, self.d, o + h
        nan = np.full(n, np.nan)
        f = {"h": np.full(n, h), "month_T": np.full(n, self.month_at(T).month),
             "month_o": np.full(n, self.mon[o]), "log_level": self.L[:, o], "d_o": d[:, o]}
        for k in (1, 2, 3, 6, 12):
            f[f"dlag{k}"] = d[:, o] - d[:, o - k] if o - k >= 0 else nan
        f["dseas"] = d[:, T - 12] - d[:, o] if 0 <= T - 12 <= o else nan
        f["dseas_prev"] = d[:, T - 13] - d[:, o] if 0 <= T - 13 <= o else nan
        f["dvol"] = np.diff(d[:, max(0, o - 6):o + 1], axis=1).std(axis=1) if o >= 2 else nan
        f["reg_d"] = self.reg[:, o]
        f["reg_dlag3"] = self.reg[:, o] - self.reg[:, o - 3] if o >= 3 else nan
        f["nat_growth_hat"] = self.N[:, o, h - 1] - self.nat[:, o]
        f["nat_yoy"] = self.nat[:, o] - self.nat[:, o - 12] if o >= 12 else nan
        if "xseries" in self.groups:
            for k in (1, 3):
                f[f"mo_all_dlag{k}"] = self.d_all[:, o] - self.d_all[:, o - k] if o >= k else nan
                f[f"mo_mean_dlag{k}"] = self.d_mo[:, o] - self.d_mo[:, o - k] if o >= k else nan
            f["reg_dlag1"] = self.reg[:, o] - self.reg[:, o - 1] if o >= 1 else nan
            f["reg_rank"] = self.reg_rank[:, o]
            f["d_minus_reg"] = d[:, o] - self.reg[:, o]
            f["nat_lag1"] = self.nat[:, o] - self.nat[:, o - 1] if o >= 1 else nan
            f["nat_seas"] = self.nat[:, T - 12] - self.nat[:, o - 12] if o >= 12 else nan
        if "calendar" in self.groups:
            cT, co = self.cal.loc[str(self.month_at(T))], self.cal.loc[str(self.month_at(o))]
            cy = self.cal.loc[str(self.month_at(T - 12))] if str(self.month_at(T - 12)) in self.cal.index else None
            f["workdays_T"] = np.full(n, cT.workdays)
            f["workdays_T_vs_o"] = np.full(n, cT.workdays - co.workdays)
            f["days_T_vs_o"] = np.full(n, cT.days - co.days)
            f["workdays_T_vs_yago"] = np.full(n, cT.workdays - cy.workdays) if cy is not None else nan
        if "fm" in self.groups:
            for nm, q in self.fm.items():
                f[f"fm_{nm}"] = q["p50"][:, o, h - 1] - d[:, o]
                f[f"fm_{nm}_iw"] = q["p90"][:, o, h - 1] - q["p10"][:, o, h - 1]
        if "news" in self.groups:
            for j, nm in enumerate(self.news_names):
                f[nm] = self.news[:, o, j]
        if "cpd" in self.groups:
            for j, nm in enumerate(self.cpd_names):
                f[nm] = self.cpd[:, o, j]
        df = pd.DataFrame(f)
        df = pd.concat([df, self.static], axis=1)
        df["sid"] = np.arange(n)
        df["o"], df["T"] = o, T
        df["w"] = self.Y[:, o]
        df["target"] = d[:, T] - d[:, o] if T < self.T else np.nan
        # full target: log y_T - log y_o = local change + national change
        df["target_full"] = self.L[:, T] - self.L[:, o] if T < self.T else np.nan
        return df


def all_rows(F: Features, min_o: int = 1) -> pd.DataFrame:
    """Feature rows for every (o, h) pair, o >= min_o, h = 1..12 (built once)."""
    X = pd.concat([F.block(o, h) for o in range(min_o, F.T) for h in range(1, HMAX + 1)],
                  ignore_index=True)
    for c in X.columns:
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)
    F._cache = {}
    return X


def fit_predict(X: pd.DataFrame, o: int, algo: str, params: dict, seed: int,
                target: str = "target", recency: float = 0.0, clip: float = 0.0):
    """Train on rows with target month <= o (and origin < o), predict rows with origin o.

    recency > 0: row weight x 0.5 ** ((o - T_row) / recency) (half-life in months).
    clip > 0: clip the training target to its [clip, 1 - clip] quantiles."""
    tr = X[(X["T"] <= o) & (X["o"] < o)]
    te = X[X["o"] == o]
    feats = [c for c in X.columns if c not in ("sid", "o", "T", "w", "target", "target_full")]
    w = tr["w"].to_numpy(np.float64)
    if recency > 0:
        w = w * 0.5 ** ((o - tr["T"].to_numpy()) / recency)
    w = pd.Series(w / w.mean(), index=tr.index)
    yt = tr[target]
    if clip > 0:
        lo, hi = yt.quantile([clip, 1 - clip])
        yt = yt.clip(lo, hi)
    if algo == "lgbm":
        import lightgbm as lgb
        p = dict(params); nr = p.pop("num_boost_round"); p["seed"] = seed
        m = lgb.train(p, lgb.Dataset(tr[feats], yt, weight=w, categorical_feature=CAT),
                      num_boost_round=nr)
        pred = m.predict(te[feats])
        imp = pd.Series(m.feature_importance("gain"), index=feats)
    else:
        from catboost import CatBoostRegressor, Pool
        Xc, Xt = tr[feats].copy(), te[feats].copy()
        for c in CAT:
            Xc[c] = Xc[c].astype(str); Xt[c] = Xt[c].astype(str)
        m = CatBoostRegressor(**{"thread_count": -1, **params}, random_seed=seed, allow_writing_files=False)
        m.fit(Pool(Xc, yt, weight=w, cat_features=CAT))
        pred = m.predict(Xt)
        imp = pd.Series(m.get_feature_importance(), index=feats)
    return te, pred, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gbm_lgb")
    ap.add_argument("--algo", choices=["lgbm", "catboost"], default="lgbm")
    ap.add_argument("--groups", nargs="+", default=["base", "calendar", "fm", "news"])
    ap.add_argument("--fm", nargs="*", default=None, help="FM names for the fm group")
    ap.add_argument("--origins", nargs="*", type=int, default=None)
    ap.add_argument("--national", choices=["v1", "v2", "v3"], default="v1")
    ap.add_argument("--target", choices=["local", "full"], default="local")
    ap.add_argument("--recency", type=float, default=0.0, help="half-life in months, 0 = off")
    ap.add_argument("--clip", type=float, default=0.0, help="target quantile clip, 0 = off")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    g = cfg["models"]["gbm_v2"]
    meta, Y, months = get_panel(cfg)
    fm_names = a.fm if a.fm is not None else [m for m in cfg["foundation_models"]["models"]]
    F = Features(cfg, meta, Y, months, a.groups, fm_names, national=a.national)
    params = dict(g["lgbm"] if a.algo == "lgbm" else g["catboost"])
    if a.threads:
        params["num_threads" if a.algo == "lgbm" else "thread_count"] = a.threads
    tcol = "target" if a.target == "local" else "target_full"
    cube = empty_cube(*Y.shape)
    origins = a.origins or list(range(2, Y.shape[1]))
    imps, t0 = [], time.time()
    X = all_rows(F)
    print(f"  rows: {len(X)}, features: {X.shape[1] - 5}, built in {time.time() - t0:.0f}s", flush=True)
    for o in origins:
        preds = []
        for s in range(a.seeds):
            Xte, pred, imp = fit_predict(X, o, a.algo, params, cfg["backtest"]["seed"] + s,
                                         tcol, a.recency, a.clip)
            preds.append(pred)
        pred = np.mean(preds, axis=0)
        hh = Xte["T"].to_numpy() - o
        sid = Xte["sid"].to_numpy()
        if a.target == "local":
            logy = F.N[sid, o, hh - 1] + F.d[sid, o] + pred
        else:
            logy = F.L[sid, o] + pred
        cube[sid, o, hh - 1] = np.exp(logy)
        imps.append(imp.rename(o))
        print(f"  {a.name} origin {o:2d}: {time.time() - t0:5.0f}s", flush=True)
    # origins 0..1: no training data -> decomposition with persistent local component
    for o in range(min(origins)):
        cube[:, o, :] = np.exp(F.N[:, o, :] + F.d[:, o:o + 1])
    save_cube(a.name, cube, groups=np.array(a.groups))
    imp = pd.concat(imps, axis=1)
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    imp.mean(axis=1).sort_values(ascending=False).to_csv(ROOT / "results" / "metrics" / f"importance_{a.name}.csv")
    print(f"{a.name} done in {time.time() - t0:.0f}s; top features:",
          list(imp.mean(axis=1).sort_values(ascending=False).index[:10]))


if __name__ == "__main__":
    main()
