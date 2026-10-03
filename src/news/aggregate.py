"""Turn GDELT GKG documents about Russia into monthly news features per municipality (МО).

Input : data/raw/gdelt/YYYYMMDD.rs.tsv.gz (src/news/gdelt_fetch.py).
Output: data/external/news_monthly_mo.parquet      territory_id x month
        data/external/news_monthly_region.parquet  region_code x month
        data/external/news_samples.csv             example URLs per МО, month, theme group
        data/external/news_location_match.csv      GDELT place -> МО match table (audit)

Geocoding. GDELT gives a place name, an ADM1 code and coordinates. GDELT geocoding of
Russian places is noisy (for example, Ukrainian towns put into Russian regions because a
Russian village has the same name). A GDELT city is matched to an МО only if
  (1) the МО centre lies within `match_radius_km` of the GDELT point, and
  (2) the transliterated МО centre name or short name starts with the same 5 letters as the
      GDELT name (Orsk -> "г Орск" -> "orsk").
Unmatched places still count at region level (through the ADM1 code).

Themes. GKG themes come from the text of each article. A document counts once per theme
group (configs/config.yaml -> news.theme_groups, matched by prefix).

Usage: .venv/bin/python src/news/aggregate.py [--workers 8]
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_config  # noqa: E402

TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
              ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
               "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))
PREFIX = re.compile(r"^(г|пгт|рп|с|п|д|ст-ца|ст|аул|х|сл|кп|дп|нп|у|м|пос|село|город)\s+", re.I)


def translit(s: str) -> str:
    s = PREFIX.sub("", str(s).strip().lower())
    return re.sub(r"[^a-z]", "", "".join(TR.get(ch, ch) for ch in s))


def key5(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())[:5]


def _groups_of(themes: str, groups: dict, exclude: set) -> list[str]:
    ts = [t for t in themes.split(";") if t and t not in exclude]
    return [g for g, prefs in groups.items() if any(t.startswith(p) for t in ts for p in prefs)]


def parse_file(args):
    """Aggregate one day: {(month, loc_key): [docs, numarts, tone_sum, neg_docs, {group: docs}]}."""
    path, groups, exclude = args
    agg = defaultdict(lambda: [0, 0, 0.0, 0, defaultdict(int)])
    locs = {}
    samples = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            month = f"{p[0][:4]}-{p[0][4:6]}"
            try:
                numarts, tone = int(p[1]), float(p[4]) if p[4] else 0.0
            except ValueError:
                continue
            gs = _groups_of(p[2], groups, exclude)
            seen = set()
            for loc in p[3].split(";"):
                q = loc.split("#")
                if len(q) < 7:
                    continue
                k = (q[0], q[1], q[3])          # type, full name, ADM1
                if k in seen:
                    continue
                seen.add(k)
                locs[k] = (float(q[4] or "nan"), float(q[5] or "nan"))
                a = agg[(month, k)]
                a[0] += 1; a[1] += numarts; a[2] += tone; a[3] += tone < -2
                for g in gs:
                    a[4][g] += 1
                if gs and q[0] == "4" and len(samples) < 400:
                    samples.append((month, k, ";".join(gs), round(tone, 2), p[5]))
    return {k: (v[0], v[1], v[2], v[3], dict(v[4])) for k, v in agg.items()}, locs, samples


def load_mo_dictionary(cfg) -> pd.DataFrame:
    d = pd.read_excel(ROOT / cfg["data"]["dictionary"])
    d = d.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    d["k_centre"] = d["municipal_district_center"].map(translit)
    d["k_short"] = d["municipal_district_name_short"].map(translit)
    return d[["territory_id", "region_code", "region_name", "municipal_district_name",
              "municipal_district_center", "k_centre", "k_short",
              "municipal_district_center_lat", "municipal_district_center_lon"]].rename(
        columns={"municipal_district_center_lat": "lat", "municipal_district_center_lon": "lon"})


def match_locations(locs: dict, mo: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    rows = []
    lat0 = np.radians(mo["lat"].to_numpy()); lon0 = np.radians(mo["lon"].to_numpy())
    kc, ks = mo["k_centre"].to_numpy(), mo["k_short"].to_numpy()
    for (typ, name, adm1), (lat, lon) in locs.items():
        r = {"type": typ, "name": name, "adm1": adm1, "lat": lat, "lon": lon, "territory_id": np.nan}
        if typ == "4" and np.isfinite(lat):
            la, lo = np.radians(lat), np.radians(lon)
            dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat0 - la) / 2) ** 2 +
                                                np.cos(la) * np.cos(lat0) * np.sin((lon0 - lo) / 2) ** 2))
            k = key5(name.split(",")[0])
            cand = np.where((dist <= radius_km) & ((np.char.startswith(kc.astype(str), k)) |
                                                   (np.char.startswith(ks.astype(str), k))))[0] if len(k) >= 4 else []
            if len(cand):
                j = cand[np.argmin(dist[cand])]
                r["territory_id"] = int(mo["territory_id"].iloc[j]); r["dist_km"] = float(dist[j])
        rows.append(r)
    m = pd.DataFrame(rows)
    # ADM1 code -> region_code by majority vote of matched cities
    mm = m.dropna(subset=["territory_id"]).merge(mo[["territory_id", "region_code"]], on="territory_id")
    adm = mm.groupby("adm1")["region_code"].agg(lambda s: s.value_counts().index[0])
    m["region_code"] = m["adm1"].map(adm)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config()
    ncfg = cfg["news"]
    files = sorted((ROOT / ncfg["gdelt_dir"]).glob("*.rs.tsv.gz"))
    print(f"{len(files)} daily files")
    groups, exclude = ncfg["theme_groups"], set(ncfg["theme_exclude"])
    total = defaultdict(lambda: [0, 0, 0.0, 0, defaultdict(int)])
    locs, samples = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (agg, lc, sm) in enumerate(ex.map(parse_file, [(f, groups, exclude) for f in files], chunksize=4)):
            for k, v in agg.items():
                t = total[k]
                t[0] += v[0]; t[1] += v[1]; t[2] += v[2]; t[3] += v[3]
                for g, c in v[4].items():
                    t[4][g] += c
            locs.update(lc); samples.extend(sm)
            if i % 100 == 0:
                print(f"  {i} files", flush=True)
    mo = load_mo_dictionary(cfg)
    match = match_locations(locs, mo, ncfg["match_radius_km"])
    ext = ROOT / "data" / "external"
    ext.mkdir(parents=True, exist_ok=True)
    match.to_csv(ext / "news_location_match.csv", index=False)
    mk = {(r.type, r.name, r.adm1): (r.territory_id, r.region_code) for r in match.itertuples()}
    rows = []
    for (month, k), v in total.items():
        tid, reg = mk.get(k, (np.nan, np.nan))
        rows.append({"month": month, "territory_id": tid, "region_code": reg, "loc_type": k[0],
                     "docs": v[0], "numarts": v[1], "tone_sum": v[2], "neg_docs": v[3],
                     **{f"g_{g}": v[4].get(g, 0) for g in groups}})
    long = pd.DataFrame(rows)
    gcols = [f"g_{g}" for g in groups]
    num = ["docs", "numarts", "tone_sum", "neg_docs"] + gcols
    mo_m = long.dropna(subset=["territory_id"]).groupby(["territory_id", "month"])[num].sum().reset_index()
    reg_m = long.dropna(subset=["region_code"]).groupby(["region_code", "month"])[num].sum().reset_index()
    mo_m.to_parquet(ext / "news_monthly_mo.parquet", index=False)
    reg_m.to_parquet(ext / "news_monthly_region.parquet", index=False)
    s = pd.DataFrame(samples, columns=["month", "loc", "groups", "tone", "url"])
    s["territory_id"] = [mk.get(k, (np.nan, np.nan))[0] for k in s["loc"]]
    s["loc"] = s["loc"].map(lambda k: k[1])
    s.dropna(subset=["territory_id"]).to_csv(ext / "news_samples.csv", index=False)
    n4 = (match.type == "4").sum()
    print(f"places: {len(match)}, cities: {n4}, matched to МО: {match.territory_id.notna().sum()} "
          f"({match.territory_id.nunique()} МО); docs matched to МО: {mo_m.docs.sum()}, "
          f"to regions: {reg_m.docs.sum()}")


if __name__ == "__main__":
    main()
