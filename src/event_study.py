"""Flood event study on a hand-checked registry of official Russian sources.

Registry: data/external/event_registry.csv (one row per event and МО / region / country, with the
announce date, the effective date and the source URL; built by hand from МЧС, regional
government and Bank of Russia pages; `verified` = the page was opened and confirms the date).

Design (floods with режим ЧС: Primorye 2023-08, Orenburg / Kurgan / Tyumen 2024-04, Omsk 2024-05):
  treated   МО named in an official flood ЧС act or report (scope = mo, verified = 1)
  controls  the other МО of the same region (no named flood); same national shock, same region
  stacking  each region has its own event month E (the most common effective month there)
  outcome   the local component d = log y - national median (the national component is removed)
  effect    difference in differences per month k relative to the event month E:
            ATT_k = mean_treated(d[E+k] - d_pre) - mean_controls(d[E+k] - d_pre),
            d_pre = mean of d over E-3..E-1. Pooled effect = mean of ATT_0..ATT_2.
  inference 95% CI: bootstrap over МО (treated and controls resampled within region), 1000 draws.
            p-value: permutation of the treated label among the МО of the same regions.
            Holm correction over the 6 categories.
  placebo   (a) the same МО and design with every event month shifted by s months, |s| >= 4
            (fake dates); (b) the permutation distribution itself (fake treated МО).
  detectors share of flood МО with an online alarm 0..2 months after E vs the base rate of
            alarms in other months ("Все категории", results/cpd/alarms_real.npz).

Outputs: results/metrics/event_study_flood.csv (per category and k), event_study_summary.csv
(pooled effect, CI, p, Holm, placebo), results/cpd/registry_events.csv (detector hit rates).

Usage: .venv/bin/python src/event_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, get_panel, load_config  # noqa: E402
from models.decomp import national_paths  # noqa: E402

REG = ROOT / "data" / "external" / "event_registry.csv"
MET = ROOT / "results" / "metrics"
CPD = ROOT / "results" / "cpd"
K = list(range(-3, 6))
POST = [0, 1, 2]
rng = np.random.default_rng(42)
warnings.filterwarnings("ignore", message="Mean of empty slice")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def did(d, tr, rid, E_of, shift=0):
    """Stacked within-region DiD. rid: region index per unit; E_of: event month per region.
    ATT_k = mean over treated units of [x_i - mean of x over the controls of its region],
    x = d[E + k] - mean(d[E-3..E-1]). Months outside the panel give NaN."""
    T = d.shape[1]
    out = []
    for k in K:
        vals = []
        for r, E in E_of.items():
            E = E + shift
            t = E + k
            m = rid == r
            if not (3 <= E and 0 <= t < T) or not (m & tr).any() or not (m & ~tr).any():
                continue
            x = d[m, t] - d[m, E - 3:E].mean(1)
            vals.append(x[tr[m]] - x[~tr[m]].mean())
        out.append(np.concatenate(vals).mean() if vals else np.nan)
    return np.array(out)


def pooled(att):
    return np.nanmean(att[[K.index(k) for k in POST]])


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    reg = pd.read_csv(REG, dtype={"territory_id": "Int64"})
    fl = reg[(reg.event_type == "flood") & (reg.scope == "mo") & (reg.verified == 1)].dropna(subset=["territory_id"])
    fl = fl.assign(E=pd.to_datetime(fl.effective_date).dt.strftime("%Y-%m"))
    fl = fl[fl.E.isin(months[3:])]
    if fl.empty:
        sys.exit("no verified flood МО in the registry")
    L = np.log(Y)
    cats = meta["category"].to_numpy()
    D = L - national_paths(L, cats)
    tids_t = set(fl.territory_id.astype(int))
    reg_t = meta.drop_duplicates("territory_id").set_index("territory_id").region_name
    fl = fl.assign(region=fl.territory_id.astype(int).map(reg_t))
    E_region = fl.groupby("region").E.agg(lambda e: months.index(e.mode().iloc[0]))
    regions = list(E_region.index)
    rows, summ = [], []
    for c in sorted(np.unique(cats)):
        m = (cats == c) & meta["region_name"].isin(regions).to_numpy()
        d = D[m]
        tid = meta.loc[m, "territory_id"].to_numpy()
        rid = np.array([regions.index(r) for r in meta.loc[m, "region_name"]])
        E_of = {regions.index(r): int(E) for r, E in E_region.items()}
        tr = np.isin(tid, list(tids_t))
        att = did(d, tr, rid, E_of)
        # bootstrap: resample treated and controls within each region
        groups = [(np.where((rid == r) & tr)[0], np.where((rid == r) & ~tr)[0]) for r in E_of]
        bs = []
        for _ in range(1000):
            ix = np.concatenate([np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))] for a, b in groups if len(a) and len(b)])
            bs.append(did(d[ix], tr[ix], rid[ix], E_of))
        bs = np.array(bs)
        for j, k in enumerate(K):
            rows.append({"spec": "main", "category": c, "rel_month": k, "att": att[j],
                         "lo95": np.nanquantile(bs[:, j], 0.025), "hi95": np.nanquantile(bs[:, j], 0.975)})
        # permutation of the treated label within regions (same count per region)
        perm = []
        for _ in range(1000):
            pt = np.zeros(len(tid), bool)
            for a, b in groups:
                pool = np.r_[a, b]
                pt[rng.choice(pool, len(a), replace=False)] = True
            perm.append(pooled(did(d, pt, rid, E_of)))
        perm = np.array(perm)
        eff = pooled(att)
        p = (np.sum(np.abs(perm) >= abs(eff)) + 1) / (len(perm) + 1)
        pb = np.array([pooled(b) for b in bs])
        # placebo dates: every event month shifted by s, |s| >= 4, while inside the panel
        plac = [pooled(did(d, tr, rid, E_of, shift=s)) for s in range(-24, 25) if abs(s) >= 4]
        plac = np.array([x for x in plac if np.isfinite(x)])
        summ.append({"category": c, "n_treated": int(tr.sum()), "n_controls": int((~tr).sum()),
                     "effect_post_0_2": eff, "lo95": np.quantile(pb, 0.025), "hi95": np.quantile(pb, 0.975),
                     "p_perm": p, "placebo_dates": len(plac), "placebo_abs_mean": np.abs(plac).mean(),
                     "placebo_share_ge_effect": (np.abs(plac) >= abs(eff)).mean()})
    out = pd.DataFrame(rows)
    s = pd.DataFrame(summ)
    s["p_holm"] = holm(s.p_perm.to_numpy())
    MET.mkdir(parents=True, exist_ok=True)
    out.to_csv(MET / "event_study_flood.csv", index=False)
    s.to_csv(MET / "event_study_summary.csv", index=False)
    pd.DataFrame({"region": regions, "event_month": [months[int(E_region[r])] for r in regions],
                  "n_treated_mo": [int(fl[fl.region == r].territory_id.nunique()) for r in regions]}
                 ).to_csv(MET / "event_study_design.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.read_csv(MET / "event_study_design.csv").to_string(index=False))
    print(s.round(4).to_string(index=False))
    print(out[out.category == "Все категории"].round(4).to_string(index=False))
    detectors(meta, months, fl)


def detectors(meta, months, fl):
    """Online alarms 0..2 months after the flood month in flood МО vs the base alarm rate."""
    z = np.load(CPD / "alarms_real.npz")
    tid = z["territory_id"]
    start = months.index(load_config()["cpd"]["start"])
    rows = []
    # online detectors run from cfg.cpd.start (2024-01): earlier events (Primorye 2023-08) cannot be hit
    ev = [(int(r.territory_id), months.index(r.E)) for r in fl.itertuples()
          if r.E in months and months.index(r.E) >= start]
    for det in [k for k in z.files if k != "territory_id"]:
        A = z[det].astype(bool)
        hits, n = 0, 0
        for t_id, e in ev:
            i = np.where(tid == t_id)[0]
            if len(i) == 0 or e + 2 >= A.shape[1]:
                continue
            n += 1
            hits += A[i[0], e:e + 3].any()
        win = np.lib.stride_tricks.sliding_window_view(A[:, start:], 3, axis=1).any(-1)
        base = win.mean()
        rows.append({"detector": det, "flood_mo": n, "hit_rate": hits / n if n else np.nan,
                     "base_rate": base, "lift": (hits / n) / base if n and base else np.nan})
    r = pd.DataFrame(rows)
    r.to_csv(CPD / "registry_events.csv", index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
