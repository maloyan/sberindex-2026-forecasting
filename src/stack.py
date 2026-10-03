"""Ensemble v2: combiners fitted on all rolling origins, chosen by blocked cross-validation.

Why v2. The v1 ensemble (src/ensemble.py) chose Caruana weights on 6 validation months
only (targets 2024-01..06). Its weights were noisy, and a single model beat it at h = 1.
v2 uses every rolling-origin forecast with a target month before the test window:

  OOF rows   all (series i, origin o, horizon h) with o >= 1 and target T = o + h <= 2024-06.
             Every base cube is causal (the model saw months 0..o only), so these rows are
             out-of-sample forecasts. Rows of the horizons h-1, h, h+1 are pooled for h.
  combiners  caruana     greedy forward selection per horizon (v1 method, more rows)
             topk_mean   mean of the K models with the lowest OOF MAE for this horizon
             topk_median median of the same K models
             lp_hc       non-negative weights that sum to 1, fitted per (horizon x
                         category) by linear programming on the OOF MAE, over the top K
                         models, then shrunk toward equal weights:  w = (1-lam) w_lp + lam/K
             + bias      optional multiplier per (horizon x category): the factor b that
                         minimises sum |y - b p| on OOF rows, shrunk toward 1 in log space
  selection  forward (v2.1, default): fit on OOF rows with target <= 2023-12, score on targets
             2024-01..06. One configuration per horizon group (h 1-6, h 7-12): the lowest
             mean score relative to the best configuration of each horizon. Horizons with
             too few rows before 2024 (h >= 11) do not vote.
             blocked (v2.0): blocked CV over target months (5 blocks of 2023-03..2024-06).
             v2.0 chose an unshrunk bias multiplier for h 7-12. The blocks of h >= 9 all
             lie in 2024-H1, so the CV could not see that a national bias of one half-year
             does not carry over; on test it raised h = 12 MAE from 658 (v1) to 818. This
             was found on the test window; see LOG.md (2026-10-03, ensemble v2).
             The test window (2024-07..12) is never used by either rule.
  outputs    results/cache/ensemble_v2.npz        fitted on all OOF rows with T <= 2024-06
             results/cache/ensemble_v2_final.npz  same configuration, refitted on T <= 2024-12
                                                   (used for the 2025 forecast)
             results/metrics/stack_cv.csv         CV MAE of every configuration
             results/metrics/stack_weights.csv    chosen weights per (horizon, category)

Usage: .venv/bin/python src/stack.py [--pool decomp|all] [--candidates a b c] [--quick] [--out name]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HMAX, ROOT, empty_cube, get_panel, load_config, save_cube  # noqa: E402
from evaluate import list_models, load_p50, windows  # noqa: E402

MET = ROOT / "results" / "metrics"
# old ensembles, raw seasonal naive, drift and tmp cubes are never candidates
NEVER_PREFIX = ("ensemble", "tmp_", "snaive", "drift")
MIN_O = 1
GROUPS = [list(range(1, 7)), list(range(7, 13))]   # horizon groups that share one configuration
FORWARD_SPLIT = 11                                  # 2023-12: forward selection fits on T <= this


def candidates_default(pool: str = "decomp_all") -> list[str]:
    """decomp_all (default, chosen by the forward score): every decomposition model with any of
    the three national versions (dec_*, dec2_*, dec3_*, gbm*, cat*, g3_loc*), so the combiner
    also mixes national forecasts.
    decomp: models built on the v3 national component only (dec3_*, gbm3_full, cat3_full).
    Raw-level models (Prophet, ETS/Theta/ARIMA, zero-shot FMs on levels, lgbm_v1, seasonal
    naive with growth) are left out: each carries its own national forecast, and in v1 the
    validation-chosen weights gave a large share to raw-level AutoTheta, which is poor on
    "Все категории". all: every cube (sensitivity)."""
    if pool == "decomp":
        return [m for m in list_models() if m.startswith("dec3_") or m in ("gbm3_full", "cat3_full")]
    if pool == "decomp_all":   # all three national versions (v1, v2, v3) and their GBMs
        return [m for m in list_models() if m.startswith(("dec_", "dec2_", "dec3_", "gbm", "cat", "g3_loc"))]
    return [m for m in list_models() if not m.startswith(NEVER_PREFIX)] + ["snaive_growth", "snaive_growth_nat",
                                                                           "snaive_growth_ba"]


class OOF:
    """Long OOF matrix for one horizon (pooled with h-1, h+1)."""

    def __init__(self, cubes: dict, Y: np.ndarray, cats: np.ndarray, h: int, T_hi: int, pool: int = 1):
        hs = [x for x in range(h - pool, h + pool + 1) if 1 <= x <= HMAX]
        pairs = [(T - x, x) for x in hs for T in range(MIN_O + 1, T_hi + 1) if T - x >= MIN_O]
        n = Y.shape[0]
        self.names = list(cubes)
        self.y = np.concatenate([Y[:, o + x] for o, x in pairs])
        self.h = np.concatenate([np.full(n, x) for o, x in pairs])
        self.T = np.concatenate([np.full(n, o + x) for o, x in pairs])
        self.cat = np.concatenate([cats for _ in pairs])
        self.M = np.stack([np.concatenate([C[:, o, x - 1] for o, x in pairs]) for C in cubes.values()]
                          ).astype(np.float64)

    def subset(self, m):
        out = OOF.__new__(OOF)
        out.names, out.y, out.T, out.h, out.cat = self.names, self.y[m], self.T[m], self.h[m], self.cat[m]
        out.M = self.M[:, m]
        return out


# ----------------------------------------------------------------------------- combiners
def caruana(M, y, steps=20):
    counts = np.zeros(len(M))
    cur = np.zeros_like(y)
    for s in range(steps):
        cand = (cur[None, :] * s + M) / (s + 1)
        j = int(np.argmin(np.abs(cand - y[None, :]).mean(axis=1)))
        counts[j] += 1
        cur = cand[j]
    return counts / counts.sum()


LP_CACHE: dict = {}


def lp_weights(M, y, max_rows=4000, seed=0):
    """argmin_w sum |y - w M|, w >= 0, sum w = 1 (exact L1 fit by linear programming)."""
    K, R = M.shape
    if R > max_rows:
        idx = np.random.default_rng(seed).choice(R, max_rows, replace=False)
        M, y, R = M[:, idx], y[idx], max_rows
    # variables: w (K), e+ (R), e- (R);  M'w + e+ - e- = y
    c = np.concatenate([np.zeros(K), np.ones(2 * R)])
    A_eq = sparse.hstack([sparse.csr_matrix(M.T), sparse.identity(R), -sparse.identity(R)])
    A_eq = sparse.vstack([A_eq, sparse.hstack([sparse.csr_matrix(np.ones((1, K))),
                                               sparse.csr_matrix((1, 2 * R))])]).tocsr()
    b_eq = np.concatenate([y, [1.0]])
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not r.success:
        return np.full(K, 1.0 / K)
    w = np.maximum(r.x[:K], 0)
    return w / w.sum()


def bias_factor(p, y):
    """argmin_b sum |y - b p| = weighted median of y/p with weights p."""
    r = y / p
    o = np.argsort(r)
    cw = np.cumsum(p[o])
    return float(r[o][np.searchsorted(cw, cw[-1] / 2)])


class Combiner:
    """cfg: dict(kind, K, lam, bias)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, D: OOF, key=None):
        c = self.cfg
        mae = np.abs(D.M - D.y[None, :]).mean(axis=1)
        self.top = np.argsort(mae)[:c.get("K", len(mae))]
        self.w = {}
        if c["kind"] == "caruana":
            w = caruana(D.M, D.y)
            self.top = np.where(w > 0)[0]
            self.w["all"] = w[self.top]
        elif c["kind"] in ("topk_mean", "topk_median"):
            self.w["all"] = np.full(len(self.top), 1.0 / len(self.top))
        elif c["kind"] == "lp_hc":
            K = len(self.top)
            for g in np.unique(D.cat):
                m = D.cat == g
                ck = (key, tuple(self.top), g)
                if key is None or ck not in LP_CACHE:
                    LP_CACHE[ck] = lp_weights(D.M[self.top][:, m], D.y[m])
                wl = LP_CACHE[ck]
                self.w[g] = (1 - c["lam"]) * wl + c["lam"] / K
        self.b = {}
        if c.get("bias", 0) > 0:
            p = self._raw(D.M, D.cat)
            for g in np.unique(D.cat):
                m = D.cat == g
                lb = np.log(bias_factor(p[m], D.y[m]))
                self.b[g] = float(np.exp((1 - c["bias_shrink"]) * lb))
        return self

    def _raw(self, M, cat):
        Mt = M[self.top]
        if self.cfg["kind"] == "topk_median":
            return np.median(Mt, axis=0)
        if "all" in self.w:
            return self.w["all"] @ Mt
        out = np.empty(M.shape[1])
        for g, w in self.w.items():
            m = cat == g
            out[m] = w @ Mt[:, m]
        return out

    def predict(self, M, cat):
        p = self._raw(M, cat)
        for g, b in self.b.items():
            p[cat == g] *= b
        return p


def configs(quick=False):
    out = [{"kind": "caruana"}, {"kind": "topk_mean", "K": 1}]
    for K in (3, 5, 8, 99):           # 99 = all candidates (equal-weight mean / median)
        out += [{"kind": "topk_mean", "K": K}, {"kind": "topk_median", "K": K}]
    for K, lam in itertools.product((5, 8), (0.0, 0.3, 0.6)):
        out.append({"kind": "lp_hc", "K": K, "lam": lam})
    with_bias = []
    for c in out:
        for s in (0.5, 0.0):
            with_bias.append({**c, "bias": 1, "bias_shrink": s})
    out += with_bias
    return out[:6] if quick else out


def name_of(c):
    s = c["kind"]
    if "K" in c:
        s += f"_K{c['K']}"
    if "lam" in c:
        s += f"_lam{c['lam']}"
    if c.get("bias"):
        s += f"_bias{1 - c['bias_shrink']:.1f}"
    return s


def blocks(T_hi):
    """5 blocks of target months in [2, T_hi]."""
    Ts = np.arange(MIN_O + 1, T_hi + 1)
    return [b for b in np.array_split(Ts, 5)]


def cv_score(D: OOF, h: int, cfg) -> float:
    """MAE on held-out target blocks, rows of horizon h only."""
    errs, n = 0.0, 0
    for bi, b in enumerate(blocks(int(D.T.max()))):
        te = np.isin(D.T, b)
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        comb = Combiner(cfg).fit(D.subset(~te), key=(h, bi))
        Dt = D.subset(te & (D.h == h))
        if len(Dt.y) == 0:
            continue
        p = comb.predict(Dt.M, Dt.cat)
        errs += np.abs(p - Dt.y).sum(); n += len(Dt.y)
    return errs / max(n, 1)


def forward_score(D: OOF, h: int, cfg, split: int = FORWARD_SPLIT) -> float:
    """Fit on OOF rows with target <= split (2023-12), score rows of horizon h with target
    in (split, 2024-06]. NaN when the fit part is too small (h >= 11): such horizons give no
    forward evidence and do not vote in the choice."""
    tr = D.T <= split
    te = (D.T > split) & (D.h == h)
    if tr.sum() < 5 * 12096 or te.sum() == 0:
        return np.nan
    comb = Combiner(cfg).fit(D.subset(tr), key=(h, "fwd"))
    Dt = D.subset(te)
    return float(np.abs(comb.predict(Dt.M, Dt.cat) - Dt.y).mean())


def apply(comb: Combiner, cubes: dict, h: int, cats: np.ndarray, shape) -> np.ndarray:
    """Forecast for horizon h at every origin."""
    n, T = shape
    out = np.full((n, T), np.nan, dtype=np.float32)
    for o in range(T):
        M = np.stack([C[:, o, h - 1] for C in cubes.values()]).astype(np.float64)
        out[:, o] = comb.predict(M, cats)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pool", choices=["decomp", "decomp_all", "all"], default="decomp_all")
    ap.add_argument("--reuse-cv", action="store_true", help="reuse results/metrics/stack_cv*.csv")
    ap.add_argument("--no-bias", action="store_true",
                    help="no (horizon x category) bias multipliers (default of v2.2, see docstring)")
    ap.add_argument("--select", choices=["forward", "blocked"], default="forward",
                    help="forward: fit on targets <= 2023-12, score 2024-01..06 (default since "
                         "v2.1); blocked: blocked CV over target months (v2.0)")
    ap.add_argument("--out", default="ensemble_v2")
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    cats = meta["category"].to_numpy()
    names = a.candidates or candidates_default(a.pool)
    print("candidates:", names)
    names = list(dict.fromkeys(names))
    cubes = {n: load_p50(n) for n in names}
    W = windows(cfg, months)
    T_val_hi, T_all_hi = max(W["validation"]), max(W["test"])
    cube_v, cube_f = empty_cube(*Y.shape), empty_cube(*Y.shape)
    cv_rows, w_rows, t0 = [], [], time.time()
    sfx = "" if a.out == "ensemble_v2" else "_" + a.out
    cv_path = MET / f"stack_cv{sfx}.csv"
    cfgs = {name_of(c): c for c in configs(a.quick) if not (a.no_bias and c.get("bias"))}
    if a.reuse_cv and cv_path.exists():
        cv = pd.read_csv(cv_path)
        cv = cv[cv.config.isin(list(cfgs))].copy()
    else:
        for h in range(1, HMAX + 1):
            D = OOF(cubes, Y, cats, h, T_val_hi)
            for nm, c in cfgs.items():
                cv_rows.append({"h": h, "config": nm, "cv_mae": cv_score(D, h, c),
                                "fwd_mae": forward_score(D, h, c)})
            print(f"  cv h={h:2d} ({time.time() - t0:.0f}s)", flush=True)
        cv = pd.DataFrame(cv_rows)
        MET.mkdir(parents=True, exist_ok=True)
        cv.to_csv(cv_path, index=False)
    # one configuration per horizon group: the lowest mean CV MAE relative to the best
    # configuration of each horizon. Choosing per single horizon among ~50 configurations
    # picked a different combiner almost every h (noise), so the choice is pooled.
    col = "fwd_mae" if a.select == "forward" else "cv_mae"
    cv["rel"] = cv[col] / cv.groupby("h")[col].transform("min")
    chosen = {}
    for grp in GROUPS:
        r = cv[cv.h.isin(grp)].dropna(subset=["rel"]).groupby("config").rel.mean().sort_values()
        print(f"h {grp[0]}-{grp[-1]}: {r.index[0]} (mean rel. CV {r.iloc[0]:.3f}); next {r.index[1]} {r.iloc[1]:.3f}")
        for h in grp:
            chosen[h] = cfgs[r.index[0]]
    for h in range(1, HMAX + 1):
        best = chosen[h]
        D = OOF(cubes, Y, cats, h, T_val_hi)
        comb = Combiner(best).fit(D)
        cube_v[:, :, h - 1] = apply(comb, cubes, h, cats, Y.shape)
        for g, w in comb.w.items():
            for j, wi in zip(comb.top, w):
                if wi > 1e-4:
                    w_rows.append({"fit_on": "T<=2024-06", "h": h, "category": g, "model": names[j],
                                   "weight": wi, "config": name_of(best)})
        for g, b in comb.b.items():
            w_rows.append({"fit_on": "T<=2024-06", "h": h, "category": g, "model": "BIAS", "weight": b,
                           "config": name_of(best)})
        Df = OOF(cubes, Y, cats, h, T_all_hi)
        combf = Combiner(best).fit(Df)
        cube_f[:, :, h - 1] = apply(combf, cubes, h, cats, Y.shape)
        for g, w in combf.w.items():
            for j, wi in zip(combf.top, w):
                if wi > 1e-4:
                    w_rows.append({"fit_on": "T<=2024-12", "h": h, "category": g, "model": names[j],
                                   "weight": wi, "config": name_of(best)})
        for g, b in combf.b.items():
            w_rows.append({"fit_on": "T<=2024-12", "h": h, "category": g, "model": "BIAS", "weight": b,
                           "config": name_of(best)})
    save_cube(a.out, cube_v)
    save_cube(f"{a.out}_final", cube_f)
    MET.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(w_rows).to_csv(MET / f"stack_weights{sfx}.csv", index=False)
    print(f"saved {a.out}, {a.out}_final in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
