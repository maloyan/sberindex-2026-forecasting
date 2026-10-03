"""Classical statistical models from statsforecast (Nixtla, Apache-2.0).

AutoETS, AutoTheta and AutoARIMA per series and origin. With at most 23 months of
history the seasonal variants rarely pass the model selection, so these are mostly
level/trend models. They are classic reference points next to Prophet.

Usage: .venv/bin/python src/models/statsf.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HMAX, empty_cube, get_panel, load_config, save_cube  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    from statsforecast import StatsForecast
    from statsforecast import models as sm
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    n, T = Y.shape
    names = cfg["models"]["statsforecast"]["models"]
    cubes = {nm: empty_cube(n, T) for nm in names}
    t0 = time.time()
    ds = pd.to_datetime(months)
    for o in range(T):
        if o + 1 < 4:
            for c in cubes.values():
                c[:, o, :] = Y[:, o:o + 1]
            continue
        df = pd.DataFrame({"unique_id": np.repeat(np.arange(n), o + 1),
                           "ds": np.tile(ds[:o + 1], n), "y": Y[:, :o + 1].ravel()})
        mods = [getattr(sm, nm)(season_length=12) for nm in names]
        sf = StatsForecast(models=mods, freq="MS", n_jobs=-1, fallback_model=sm.Naive())
        fc = sf.forecast(df=df, h=HMAX)
        fc = fc.sort_values(["unique_id", "ds"])
        for nm in names:
            cubes[nm][:, o, :] = np.maximum(fc[nm].to_numpy().reshape(n, HMAX), 0)
        print(f"  statsforecast origin {o}: {time.time() - t0:.0f}s", flush=True)
    for nm, c in cubes.items():
        save_cube(nm.lower(), c)
    print("done")


if __name__ == "__main__":
    main()
