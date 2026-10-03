"""Change-point benchmark: synthetic injection with equal false-alarm calibration + real cases.

1. Synthetic. Take real local components d of category "Все категории" (no injected
   break yet) and inject, at a random month tau in 2024-02..2024-10:
     shift  a log level step of size 5/10/20/30% (random sign)
     trend  a slope change of 2/4% per month
     spike  a one-month outlier of 20% (not a structural break: a good detector ignores it)
   Each online detector has one threshold parameter. It is calibrated on half of the
   series (the "calibration" half) so that its alarm rate on clean series is close to
   `target_far` alarms per series-month. All metrics are reported on the other half.
   Metrics: recall within `tolerance_months`, mean delay, precision, F1, false-alarm rate
   on clean series, alarm rate on spikes.
2. Offline detectors on synthetic shifts: F1 with tolerance, mean absolute position error.
3. Real data. Online detectors on all МО, 2024; the Orsk flood case (2024-04); a news-
   labelled event set (МО-months with a spike of flood/disaster news in GDELT): the hit
   rate of each detector within 0..2 months after a news event vs its base alarm rate
   (lift). The news-aware BOCPD raises the hazard in months with a news spike.
4. Break-aware forecasting: on synthetic shifts, the local seasonal forecaster with and
   without the adjustment of the pre-break history by the detected jump.

Usage: .venv/bin/python src/cpd/benchmark.py [--n-series 400]
"""
from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detectors as D  # noqa: E402
from common import ROOT, get_panel, load_config  # noqa: E402
from models.decomp import national_paths  # noqa: E402

OUT = ROOT / "results" / "cpd"
GRIDS = {
    "spike": ("k", [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]),
    "cusum": ("h", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
    "pelt_online": ("penalty", [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
    "binseg_online": ("penalty", [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
    "window_online": ("penalty", [4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]),
    "bocpd": ("threshold", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "bocpd_confirm": ("threshold", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "resid_cusum": ("h", [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
}


def local_seasonal_resid(d):
    """One-step error of the local seasonal-naive forecaster: d_t - (d_{t-1} + d_{t-12} - d_{t-13})."""
    e = np.full_like(d, np.nan)
    e[:, 13:] = d[:, 13:] - d[:, 12:-1] - d[:, 1:-12] + d[:, :-13]
    e[:, 1:13] = d[:, 1:13] - d[:, :12]
    return e


def run_online(name, d, start, param):
    z, _ = D.standardize(d)
    if name == "spike":
        return D.spike(z, start, k=param)
    if name == "cusum":
        return D.cusum(z, start, h=param)
    if name == "pelt_online":
        return D.pelt_online(z, start, penalty=param)
    if name == "binseg_online":
        return D.binseg_online(z, start, penalty=param)
    if name == "window_online":
        return D.window_online(z, start, penalty=param)
    if name == "bocpd":
        return D.bocpd(z, start, threshold=param)[0]
    if name == "bocpd_confirm":
        return D.bocpd(z, start, threshold=param, lo=2, short=3)[0]
    if name == "resid_cusum":
        return D.resid_cusum(local_seasonal_resid(d), start, h=param)
    raise ValueError(name)


def score(A, tau, tol, start):
    """Event metrics. A hit = first alarm in [tau, tau+tol]. Other alarm episodes = false."""
    hits, delays, fp = [], [], 0
    for i in range(A.shape[0]):
        a = np.where(A[i, start:])[0] + start
        ok = a[(a >= tau[i]) & (a <= tau[i] + tol)]
        hits.append(len(ok) > 0)
        if len(ok):
            delays.append(ok[0] - tau[i])
        bad = a[(a < tau[i]) | (a > tau[i] + tol)]
        # count episodes (consecutive alarms = one episode)
        fp += int((np.diff(bad, prepend=-10) > 1).sum()) if len(bad) else 0
    tp = int(np.sum(hits))
    rec = tp / len(hits)
    prec = tp / (tp + fp) if tp + fp else np.nan
    return {"recall": rec, "precision": prec,
            "F1": 2 * prec * rec / (prec + rec) if prec and rec else 0.0,
            "mean_delay": float(np.mean(delays)) if delays else np.nan}


def inject(d, kind, size, tau, sign):
    x = d.copy()
    T = d.shape[1]
    for j in range(len(x)):
        t0 = tau[j]
        if kind == "shift":
            x[j, t0:] += sign[j] * size
        elif kind == "trend":
            x[j, t0:] += sign[j] * size * np.arange(1, T - t0 + 1)
        elif kind == "spike":
            x[j, t0] += sign[j] * size
    return x


def calibrate(name, d_clean, start, target_far):
    pname, grid = GRIDS[name]
    best, rows = None, []
    for p in grid:
        far = run_online(name, d_clean, start, p)[:, start:].mean()
        rows.append((p, far))
    # the most sensitive setting whose clean alarm rate does not exceed the target
    # every parameter is "larger = stricter": take the most sensitive one within the target
    ok = [p for p, f in rows if f <= target_far]
    best = min(ok) if ok else max(grid)
    return best, rows


def synthetic(cfg, d_all, rng, n_series, target_far=0.02):
    c = cfg["cpd"]; syn = c["synthetic"]; start = 12; tol = c["tolerance_months"]
    pick = rng.choice(len(d_all), size=min(n_series, len(d_all)), replace=False)
    half = len(pick) // 2
    cal, ev = d_all[pick[:half]], d_all[pick[half:]]
    rows, params = [], {}
    for name in GRIDS:
        t0 = time.time()
        p, curve = calibrate(name, cal, start, target_far)
        params[name] = p
        far = run_online(name, ev, start, p)[:, start:].mean()
        scen = [("shift", s) for s in syn["shift_sizes"]] + [("trend", s) for s in syn["trend_slopes"]] \
            + [("spike", s) for s in syn["spike_sizes"]]
        for kind, size in scen:
            r = np.random.default_rng(zlib.crc32(f"{kind}{size}".encode()))
            tau = r.integers(13, 22, size=len(ev)); sign = r.choice([-1, 1], size=len(ev))
            A = run_online(name, inject(ev, kind, size, tau, sign), start, p)
            s = score(A, tau, tol, start)
            rows.append({"detector": name, "param": f"{GRIDS[name][0]}={p}", "scenario": kind,
                         "size": size, **s, "clean_FAR": far})
        print(f"  {name}: param {p}, clean FAR {far:.3f}, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), params, pick


def offline_synthetic(cfg, d_all, pick, tol=1):
    rows = []
    ev = d_all[pick[len(pick) // 2:]]
    for size in cfg["cpd"]["synthetic"]["shift_sizes"]:
        r = np.random.default_rng(int(size * 1000))
        tau = r.integers(6, 19, size=len(ev)); sign = r.choice([-1, 1], size=len(ev))
        x = inject(ev, "shift", size, tau, sign)
        z, _ = D.standardize(x)
        for m in ("pelt", "binseg", "bottomup", "window", "kernel", "bocpd_map"):
            bks = D.offline(z, m, penalty=cfg["cpd"]["offline"]["pelt_penalty"] / 2)
            tp = fp = 0; err = []
            for b, t in zip(bks, tau):
                near = [bb for bb in b if abs(bb - t) <= tol]
                tp += bool(near); fp += len(b) - bool(near)
                if near:
                    err.append(min(abs(bb - t) for bb in near))
            rec = tp / len(tau); prec = tp / (tp + fp) if tp + fp else np.nan
            rows.append({"method": m, "size": size, "recall": rec, "precision": prec,
                         "F1": 2 * prec * rec / (prec + rec) if prec and rec else 0,
                         "MAE_position": np.mean(err) if err else np.nan,
                         "breaks_per_series": np.mean([len(b) for b in bks])})
    return pd.DataFrame(rows)


def news_semi_synthetic(cfg, d_all, pick, params, p_news_break=0.5, p_news_other=0.03, size=0.10):
    """News-aware detectors when news comes with a share of the true breaks.

    A news flag is set at the break month with probability p_news_break and in any other
    month with probability p_news_other (noise). The base rate of news months in the real
    GDELT data is close to p_news_other. The news-aware CUSUM lowers its threshold to 60%
    in a news month and the next one; the news-aware BOCPD multiplies the hazard by 5.
    """
    ev = d_all[pick[len(pick) // 2:]]
    r = np.random.default_rng(7)
    tau = r.integers(13, 22, size=len(ev)); sign = r.choice([-1, 1], size=len(ev))
    x = inject(ev, "shift", size, tau, sign)
    rows = []
    for with_break in (True, False):
        xx = x if with_break else ev
        news = r.random(xx.shape) < p_news_other
        if with_break:
            news[np.arange(len(ev)), tau] |= r.random(len(ev)) < p_news_break
        hs = np.ones(xx.shape); hs[:, 1:] += 4 * (news[:, :-1] | news[:, 1:]); hs[:, 0] += 4 * news[:, 0]
        z, _ = D.standardize(xx)
        al = {"cusum": D.cusum(z, 12, h=params["cusum"]),
              "cusum_news": D.cusum(z, 12, h=params["cusum"], thr_scale=np.where(hs > 1, 0.6, 1.0)),
              "bocpd": D.bocpd(z, 12, threshold=params["bocpd"])[0],
              "bocpd_news": D.bocpd(z, 12, threshold=params["bocpd"], hazard_scale=hs)[0]}
        for nm, A in al.items():
            if with_break:
                rows.append({"detector": nm, **score(A, tau, cfg["cpd"]["tolerance_months"], 12)})
            else:
                rows[-4 + list(al).index(nm)]["clean_FAR"] = A[:, 12:].mean()
    return pd.DataFrame(rows)


def localise(x, alarm, o, back=3, win=8):
    """Break month = the split in [alarm-back, alarm] with the lowest two-mean squared error."""
    lo = max(1, o - win)
    best, bb = np.inf, alarm
    for b in range(max(lo + 1, alarm - back), alarm + 1):
        a, c = x[lo:b], x[b:o + 1]
        sse = ((a - a.mean()) ** 2).sum() + ((c - c.mean()) ** 2).sum()
        if sse < best:
            best, bb = sse, b
    return bb


def break_aware_demo(cfg, d_all, Y_all, nat_all, pick, params, rng, size=0.15):
    """Forecast after a detected break: local seasonal model with vs without adjustment."""
    ev = pick[len(pick) // 2:]
    d = d_all[ev]
    tau = rng.integers(13, 19, size=len(ev)); sign = rng.choice([-1, 1], size=len(ev))
    x = inject(d, "shift", size, tau, sign)
    A = run_online("cusum", x, 12, params["cusum"])
    rows = []
    for j in range(len(ev)):
        al = np.where(A[j] & (np.arange(x.shape[1]) >= tau[j]))[0]
        if not len(al):
            continue
        b_alarm = al[0]
        for o in range(b_alarm, min(b_alarm + 3, 22)):
            for h in (1, 2, 3, 6):
                T = o + h
                if T > 23:
                    continue
                truth = x[j, T]
                kk = 3
                # local seasonal model with the YoY drift averaged over k = 3 months
                plain = x[j, T - 12] + np.mean([x[j, o - i] - x[j, o - i - 12] for i in range(kk)])
                # break-aware: the alarm comes with a delay, so the break month is first
                # localised (best two-mean split of months o-8..o, break in [alarm-3, alarm]);
                # then the pre-break months are shifted by the estimated jump
                bhat = localise(x[j], b_alarm, o)
                J = x[j, bhat:o + 1].mean() - x[j, max(0, bhat - 4):bhat].mean()
                adj = lambda t: x[j, t] + (J if t < bhat else 0.0)  # noqa: E731
                aware = adj(T - 12) + np.mean([adj(o - i) - adj(o - i - 12) for i in range(kk)])
                persist = x[j, o]
                lvl = np.exp(nat_all[ev[j], T])
                for nm, p in (("seasonal_k3", plain), ("seasonal_k3_break_aware", aware),
                              ("persistence", persist)):
                    rows.append({"model": nm, "h": h, "abs_err_rub": abs(np.exp(p) - np.exp(truth)) * lvl})
    r = pd.DataFrame(rows)
    return r.groupby(["model", "h"]).abs_err_rub.mean().unstack().round(1)


def real_cases(cfg, meta_c, d, months, params):
    start = 12
    names = list(GRIDS)
    alarms = {nm: run_online(nm, d, start, params[nm]) for nm in names}
    out = {}
    # news-labelled events
    news_p = ROOT / "data" / "external" / "news_monthly_mo.parquet"
    if news_p.exists():
        nw = pd.read_parquet(news_p)
        nw["ev"] = nw["g_flood"] + nw["g_disaster"]
        piv = nw.pivot_table(index="territory_id", columns="month", values="ev", aggfunc="sum").reindex(columns=months).fillna(0)
        E = piv.reindex(meta_c["territory_id"]).fillna(0).to_numpy()
        # surprise: count vs the mean of the previous 6 months, and at least 10 documents
        prev = pd.DataFrame(E).T.rolling(6, min_periods=3).mean().shift(1).T.to_numpy()
        ev_mask = (E >= 10) & (E > 3 * np.nan_to_num(prev, nan=0) + 2)
        ev_mask[:, :start] = False
        ev_mask[:, -2:] = False             # need 2 months after the event
        # news-aware BOCPD: hazard x (1 + 4) in an event month and the next one
        hs = np.ones_like(E, dtype=float)
        hs[:, 1:] += 4 * (ev_mask[:, :-1] | ev_mask[:, 1:])
        z, _ = D.standardize(d)
        hs[:, 0] += 4 * ev_mask[:, 0]
        alarms["bocpd_news"] = D.bocpd(z, start, threshold=params["bocpd"], hazard_scale=hs)[0]
        alarms["cusum_news"] = D.cusum(z, start, h=params["cusum"], thr_scale=np.where(hs > 1, 0.6, 1.0))
        rows = []
        idx_i, idx_t = np.where(ev_mask)
        for nm, A in alarms.items():
            hit = np.mean([A[i, t:t + 3].any() for i, t in zip(idx_i, idx_t)]) if len(idx_i) else np.nan
            base = np.mean([A[:, t:t + 3].any(axis=1).mean() for t in range(start, A.shape[1] - 2)])
            rows.append({"detector": nm, "news_events": len(idx_i), "hit_rate_0_2m": hit,
                         "base_rate_0_2m": base, "lift": hit / base if base else np.nan,
                         "alarm_rate": A[:, start:].mean()})
        out["news_events"] = pd.DataFrame(rows)
        out["news_event_list"] = pd.DataFrame({"territory_id": meta_c["territory_id"].to_numpy()[idx_i],
                                               "name": meta_c["name"].to_numpy()[idx_i],
                                               "month": np.array(months)[idx_t],
                                               "news_docs": E[idx_i, idx_t]})
    # offline PELT on all real series: breaks by month (data artefacts show up as pile-ups)
    zr, _ = D.standardize(d)
    bks = D.offline(zr, "pelt", penalty=cfg["cpd"]["offline"]["pelt_penalty"], min_size=cfg["cpd"]["offline"]["min_size"])
    cnt = pd.Series([months[b] for bk in bks for b in bk]).value_counts().reindex(months, fill_value=0)
    out["offline_pelt_breaks_by_month"] = cnt.rename("breaks").to_frame()
    # alarms by month and the Orsk case
    out["alarms_by_month"] = pd.DataFrame({nm: A[:, start:].sum(0) for nm, A in alarms.items()},
                                          index=months[start:])
    case_ids = {"Орск": 1673}
    rows = []
    for nm_case, tid in case_ids.items():
        i = np.where(meta_c["territory_id"].to_numpy() == tid)[0]
        if not len(i):
            continue
        i = i[0]
        for t in range(start, d.shape[1]):
            rows.append({"case": nm_case, "month": months[t], "d": d[i, t],
                         **{nm: bool(A[i, t]) for nm, A in alarms.items()}})
    out["cases"] = pd.DataFrame(rows)
    return out, alarms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=cfg_default("n_series"))
    ap.add_argument("--target-far", type=float, default=0.02)
    a = ap.parse_args()
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    L = np.log(Y)
    nat = national_paths(L, meta["category"].to_numpy())
    sel = (meta["category"] == cfg["cpd"]["category"]).to_numpy()
    meta_c = meta[sel].reset_index(drop=True)
    d = (L - nat)[sel]
    rng = np.random.default_rng(cfg["backtest"]["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    print("== synthetic, online detectors (equal false-alarm calibration) ==")
    syn, params, pick = synthetic(cfg, d, rng, a.n_series, a.target_far)
    syn.to_csv(OUT / "synthetic_online.csv", index=False)
    print(syn.pivot_table(index="detector", columns=["scenario", "size"], values="F1").round(2).to_string())
    print(syn.pivot_table(index="detector", columns=["scenario", "size"], values="mean_delay").round(2).to_string())

    print("== synthetic, offline detectors ==")
    off = offline_synthetic(cfg, d, pick)
    off.to_csv(OUT / "synthetic_offline.csv", index=False)
    print(off.pivot(index="method", columns="size", values="F1").round(2).to_string())

    print("== news-aware detectors, semi-synthetic (10% shifts, news on half of the breaks) ==")
    ns = news_semi_synthetic(cfg, d, pick, params)
    ns.to_csv(OUT / "news_semi_synthetic.csv", index=False)
    print(ns.round(3).to_string(index=False))

    print("== break-aware forecasting (synthetic 15% shift), MAE rub ==")
    ba = break_aware_demo(cfg, d, Y[sel], nat[sel], pick, params, rng)
    ba.to_csv(OUT / "break_aware_synthetic.csv")
    print(ba.to_string())

    print("== real data ==")
    real, alarms = real_cases(cfg, meta_c, d, months, params)
    for k, v in real.items():
        v.to_csv(OUT / f"real_{k}.csv", index=k in ("alarms_by_month", "offline_pelt_breaks_by_month"))
        print(f"-- {k}"); print(v.head(30).round(3).to_string())
    np.savez_compressed(OUT / "alarms_real.npz", **alarms, territory_id=meta_c["territory_id"].to_numpy())
    pd.Series(params).to_csv(OUT / "calibrated_params.csv")


def cfg_default(key):
    return load_config()["cpd"]["synthetic"][key]


if __name__ == "__main__":
    main()
