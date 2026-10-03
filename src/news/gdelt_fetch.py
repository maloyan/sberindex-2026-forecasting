"""Download GDELT 1.0 daily GKG files and keep only documents about places in Russia.

GDELT (https://www.gdeltproject.org) is free for any use, with a citation
("The GDELT Project, https://www.gdeltproject.org"). The Global Knowledge Graph (GKG)
holds themes, locations and tone that GDELT extracts from the text of news articles.

For each day the script streams the zip, keeps rows whose LOCATIONS field has a
Russian city (type 4) or Russian region (type 5) and writes a gzip TSV with the
columns: DATE, NUMARTS, THEMES, LOCATIONS_RS, TONE, SOURCEURL.

Only the Python standard library is used, so the script runs on any machine.

Usage:
  python src/news/gdelt_fetch.py --start 2022-12-01 --end 2024-12-31 --out data/raw/gdelt --workers 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

URL = "https://data.gdeltproject.org/gkg/{d}.gkg.csv.zip"


def keep_rs(locations: str) -> str:
    """Return only the Russian city/region entries of a GKG 1.0 LOCATIONS field."""
    out = []
    for loc in locations.split(";"):
        p = loc.split("#")
        if len(p) >= 7 and p[2] == "RS" and p[0] in ("4", "5"):
            out.append(loc)
    return ";".join(out)


def one_day(day: str, out_dir: str, retries: int = 4) -> str:
    dst = os.path.join(out_dir, f"{day}.rs.tsv.gz")
    if os.path.exists(dst):
        return f"{day} skip"
    for a in range(retries):
        try:
            with urllib.request.urlopen(URL.format(d=day), timeout=180) as r:
                blob = r.read()
            break
        except Exception as e:  # network errors: retry with backoff
            if a == retries - 1:
                return f"{day} FAIL {e}"
            time.sleep(5 * (a + 1))
    n_in = n_out = 0
    tmp = dst + ".part"
    with zipfile.ZipFile(io.BytesIO(blob)) as z, gzip.open(tmp, "wt", encoding="utf-8") as w:
        name = z.namelist()[0]
        with z.open(name) as f:
            for i, raw in enumerate(io.TextIOWrapper(f, encoding="utf-8", errors="replace")):
                if i == 0:
                    w.write("DATE\tNUMARTS\tTHEMES\tLOCATIONS_RS\tTONE\tSOURCEURL\n")
                    continue
                n_in += 1
                if "#RS#" not in raw:
                    continue
                c = raw.rstrip("\n").split("\t")
                if len(c) < 11:
                    continue
                locs = keep_rs(c[4])
                if not locs:
                    continue
                tone = c[7].split(",")[0] if c[7] else ""
                w.write("\t".join([c[0], c[1], c[3], locs, tone, c[10].split("<UDIV>")[0]]) + "\n")
                n_out += 1
    os.replace(tmp, dst)
    return f"{day} in={n_in} rs={n_out}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-12-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default="data/raw/gdelt")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d0, d1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [(d0 + dt.timedelta(k)).strftime("%Y%m%d") for k in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda d: one_day(d, a.out), days):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
