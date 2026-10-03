"""The first baseline (src/forecast_baseline.py): global LightGBM on log(y_T / y_o), as a cube.

Kept to show the effect of the later changes (decomposition, FM features, news). It is
trained per origin on pairs with target <= origin, like every other model here.

Usage: .venv/bin/python src/models/lgbm_v1.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HMAX, empty_cube, get_panel, load_config, save_cube  # noqa: E402
from forecast_baseline import lgbm_fit_predict  # noqa: E402


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    cube = empty_cube(*Y.shape)
    t0 = time.time()
    for o in range(Y.shape[1]):
        if o < 4:
            cube[:, o, :] = Y[:, o:o + 1]
            continue
        pairs = [(o, h) for h in range(1, HMAX + 1)]
        Xte, _ = lgbm_fit_predict(Y, meta, months, o, pairs, cfg["models"]["lgbm_v1"], seed=cfg["backtest"]["seed"])
        cube[Xte["sid"].to_numpy(), o, Xte["h"].to_numpy() - 1] = Xte["pred"].to_numpy()
        print(f"  lgbm_v1 origin {o}: {time.time() - t0:.0f}s", flush=True)
    save_cube("lgbm_v1", cube)


if __name__ == "__main__":
    main()
