"""Figures for the report and the presentation (report/figures/*.png).

Usage: .venv/bin/python src/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, get_panel, load_config, load_cube  # noqa: E402
from models.decomp import national_paths  # noqa: E402

FIG = ROOT / "report" / "figures"
MET = ROOT / "results" / "metrics"
CPD = ROOT / "results" / "cpd"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

NAMES = {
    "naive": "Наивная", "snaive": "Сезонная наивная", "snaive_growth": "Сез. наивная + рост",
    "snaive_growth_nat": "Сез. наивная + нац. рост", "drift": "Дрейф",
    "prophet": "Prophet (базовая)", "autoets": "ETS", "autotheta": "Theta", "autoarima": "ARIMA",
    "lgbm_v1": "LightGBM v1", "dec_persist": "Декомпозиция", "dec_mean3": "Декомп. (ср. 3 мес.)",
    "dec_seas": "Декомп. (лок. сезон)", "gbm_base": "GBM база", "gbm_news": "GBM + новости",
    "gbm_cpd": "GBM + CPD", "gbm_fm": "GBM + FM", "gbm_full": "GBM: всё (LightGBM)",
    "cat_full": "GBM: всё (CatBoost)", "ensemble": "Ансамбль",
}
FM = {"chronos_bolt_small": "Chronos-Bolt S", "chronos_bolt_base": "Chronos-Bolt B",
      "chronos2": "Chronos-2", "chronos2_mv": "Chronos-2 (мульти)", "chronos2_ft": "Chronos-2 дообуч.",
      "timesfm25": "TimesFM 2.5", "tirex2": "TiRex-2", "toto2_22m": "Toto-2 22M", "toto2_313m": "Toto-2 313M"}
for k, v in FM.items():
    NAMES[f"fm_{k}"] = f"{v} (уровни)"
    NAMES[f"dec_fmlocal_{k}"] = f"{v} (декомп.)"

C_MAIN, C_PROPHET, C_GREY, C_FM = "#2b6cb0", "#c53030", "#a0aec0", "#6b46c1"


def nm(m):
    return NAMES.get(m, m)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


def fig_national(meta, Y, months):
    L = np.log(Y)
    nat = national_paths(L, meta["category"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = pd.to_datetime(months)
    for c in sorted(meta["category"].unique()):
        i = np.where(meta["category"].to_numpy() == c)[0][0]
        ax.plot(x, np.exp(nat[i] - nat[i, 0]) * 100, label=c, lw=1.8)
    ax.set_ylabel("индекс, январь 2023 = 100")
    ax.set_title("Национальная компонента: медиана по МО (сезонность и инфляция общие для всех)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    save(fig, "fig_national.png")


def fig_mae_h(test, key):
    t = test[test.model.isin(key)].pivot(index="model", columns="h", values="MAE").reindex(key).dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    H = list(t.columns)
    w = 0.8 / len(t)
    for j, (m, row) in enumerate(t.iterrows()):
        col = C_PROPHET if m == "prophet" else (C_MAIN if m == "ensemble" else
                                                (C_FM if "fm" in m else C_GREY))
        ax.bar(np.arange(len(H)) + j * w, row.values, w, label=nm(m), color=col,
               alpha=1 if m in ("prophet", "ensemble") else 0.75, edgecolor="white")
    ax.set_xticks(np.arange(len(H)) + 0.4 - w / 2, [f"h = {h}" for h in H])
    ax.set_ylabel("MAE, руб. на жителя в месяц")
    ax.set_title("Отложенное окно (test, 2024-07..12): MAE по горизонтам")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    save(fig, "fig_mae_by_h_test.png")


def fig_curve(key):
    c = pd.read_csv(MET / "mae_curve_test.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for m in key:
        if m in c.index:
            ax.plot(c.columns.astype(int), c.loc[m], marker="o", ms=3, label=nm(m),
                    lw=2.4 if m in ("ensemble", "prophet") else 1.2,
                    color=C_PROPHET if m == "prophet" else (C_MAIN if m == "ensemble" else None))
    ax.set_xlabel("горизонт h, месяцев"); ax.set_ylabel("MAE, руб.")
    ax.set_title("MAE на test для каждого горизонта 1..12")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    save(fig, "fig_mae_curve.png")


def fig_fm(test):
    rows = []
    for k, v in FM.items():
        for space, pre in (("уровни (как есть)", "fm_"), ("локальная компонента", "dec_fmlocal_")):
            m = test[test.model == pre + k]
            for _, r in m.iterrows():
                rows.append({"fm": v, "space": space, "h": r.h, "MAE": r.MAE})
    d = pd.DataFrame(rows)
    if d.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=False)
    pr = test[test.model == "prophet"].set_index("h").MAE
    for ax, h in zip(axes, (1, 3, 6, 12)):
        s = d[d.h == h].pivot(index="fm", columns="space", values="MAE")
        s = s.reindex([v for v in FM.values() if v in s.index])
        y = np.arange(len(s))
        if "уровни (как есть)" in s:
            ax.barh(y + 0.2, s["уровни (как есть)"], 0.4, color=C_GREY, label="уровни (как есть)")
        if "локальная компонента" in s:
            ax.barh(y - 0.2, s["локальная компонента"], 0.4, color=C_FM, label="локальная компонента")
        if h in pr.index:
            ax.axvline(pr[h], color=C_PROPHET, ls="--", lw=1.2, label="Prophet")
        ax.set_yticks(y, s.index if h == 1 else [""] * len(s), fontsize=8)
        ax.set_title(f"h = {h}"); ax.set_xlabel("MAE, руб.")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Фундаментальные модели zero-shot: сырые уровни против локальной компоненты (test)", y=1.04)
    save(fig, "fig_fm_level_vs_local.png")


def fig_bootstrap(key):
    p = MET / "bootstrap_test_vs_prophet.csv"
    if not p.exists():
        return
    b = pd.read_csv(p)
    b = b[b.model.isin(key)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    order = [m for m in key if m in set(b.model)]
    for ax, h in zip(axes, (1, 3, 6, 12)):
        s = b[b.h == h].set_index("model").reindex(order)
        y = np.arange(len(s))
        ax.errorbar(s.dMAE, y, xerr=[s.dMAE - s.lo95, s.hi95 - s.dMAE], fmt="o", ms=4,
                    color=C_MAIN, ecolor=C_GREY, capsize=2)
        ax.axvline(0, color=C_PROPHET, lw=1)
        ax.set_title(f"h = {h}"); ax.set_xlabel("MAE модели − MAE Prophet, руб.")
        ax.set_yticks(y, [nm(m) for m in s.index], fontsize=8)
    fig.suptitle("Разница с Prophet и 95% доверительный интервал (кластерный бутстреп по МО)", y=1.04)
    save(fig, "fig_bootstrap_vs_prophet.png")


def fig_cpd():
    p = CPD / "synthetic_online.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s["col"] = s.scenario.map({"shift": "сдвиг", "trend": "тренд", "spike": "выброс"}) + " " + \
        (s["size"] * 100).round().astype(int).astype(str) + "%"
    piv = s.pivot_table(index="detector", columns="col", values="F1")
    order = s.drop_duplicates("col").sort_values(["scenario", "size"], key=lambda c: c.map({"shift": 0, "trend": 1, "spike": 2}) if c.name == "scenario" else c)
    cols = [c for c in order.col if not c.startswith("выброс")]
    piv_sp = s[s.scenario == "spike"].set_index("detector")["recall"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    im = axes[0].imshow(piv[cols].values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(piv)), piv.index, fontsize=8)
    for i in range(len(piv)):
        for j, c in enumerate(cols):
            axes[0].text(j, i, f"{piv[c].iloc[i]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if piv[c].iloc[i] > 0.6 else "black")
    axes[0].grid(False)
    axes[0].set_title("F1 (допуск 2 мес.) при одинаковой частоте ложных тревог ≈2%")
    fig.colorbar(im, ax=axes[0], fraction=0.03)
    sp = piv_sp.reindex(piv.index)
    axes[1].barh(range(len(sp)), sp.values, color=C_PROPHET, alpha=0.7)
    axes[1].set_yticks(range(len(sp)), [""] * len(sp)); axes[1].invert_yaxis()
    axes[0].set_ylim(len(piv) - 0.5, -0.5)
    axes[1].set_title("Тревога на выбросе 20%\n(меньше = лучше)", fontsize=9)
    axes[1].set_xlim(0, 1)
    save(fig, "fig_cpd_synthetic.png")


def fig_orsk(meta, Y, months):
    tid = 1673
    sel = (meta.territory_id == tid).to_numpy() & (meta.category == "Все категории").to_numpy()
    if not sel.any():
        return
    i = np.where(sel)[0][0]
    L = np.log(Y); nat = national_paths(L, meta["category"].to_numpy()); d = L - nat
    x = pd.to_datetime(months)
    news = pd.read_parquet(ROOT / "data" / "external" / "news_monthly_mo.parquet")
    nw = news[news.territory_id == tid].set_index("month").reindex(months).fillna(0)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(x, (np.exp(d[i]) - 1) * 100, color=C_MAIN, marker="o", ms=3, label="Орск, отклонение от национальной компоненты")
    al = CPD / "real_cases.csv"
    if al.exists():
        c = pd.read_csv(al)
        for det, mk, col in (("cusum", "v", "#dd6b20"), ("cusum_news", "*", C_PROPHET), ("bocpd", "s", C_FM)):
            if det in c:
                mm = c[c[det]].month
                xi = [months.index(m) for m in mm]
                if not xi:
                    continue
                ax[0].scatter(x[xi], (np.exp(d[i, xi]) - 1) * 100 + 2, marker=mk, s=60, color=col, label=f"тревога: {det}")
    ax[0].axvline(pd.Timestamp("2024-04-05"), color="grey", ls=":", lw=1)
    ax[0].text(pd.Timestamp("2024-03-28"), ax[0].get_ylim()[0] + 1, "прорыв дамбы\n5 апреля 2024", fontsize=8, ha="right")
    ax[0].set_ylabel("% к национальному уровню"); ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[0].set_title("Орск (Оренбургская обл.): паводок апреля 2024 в данных и в новостях")
    ax[1].bar(x, nw["g_flood"], width=20, color="#3182ce", label="документы GDELT с темой «наводнение»")
    ax[1].bar(x, nw["docs"] - nw["g_flood"], bottom=nw["g_flood"], width=20, color=C_GREY, label="прочие документы")
    ax[1].set_yscale("symlog"); ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_ylabel("документов")
    save(fig, "fig_orsk.png")


def fig_examples(meta, Y, months):
    try:
        E = load_cube("ensemble"); P = load_cube("prophet")
    except FileNotFoundError:
        return
    o = months.index("2024-06")
    picks = [(1673, "Все категории"), (1673, "Продовольствие"), (2210, "Все категории"), (10, "Маркетплейсы")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    x = pd.to_datetime(months)
    for ax, (tid, cat) in zip(axes.ravel(), picks):
        s = np.where((meta.territory_id == tid).to_numpy() & (meta.category == cat).to_numpy())[0]
        if not len(s):
            continue
        s = s[0]
        ax.plot(x, Y[s], color="black", lw=1.4, label="факт")
        xf = x[o + 1:o + 7]
        ax.plot(xf, E[s, o, :6], color=C_MAIN, marker="o", ms=3, label="ансамбль")
        ax.plot(xf, P[s, o, :6], color=C_PROPHET, ls="--", label="Prophet")
        ax.axvline(x[o], color="grey", ls=":")
        ax.set_title(f"{meta.name.iloc[s][:40]}\n{cat}", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.autofmt_xdate()
    fig.suptitle("Прогноз из origin 2024-06 на 1..6 месяцев (руб. на жителя)")
    fig.tight_layout()
    save(fig, "fig_examples.png")


def fig_ablation(test):
    ab = ["gbm_base", "gbm_news", "gbm_cpd", "gbm_fm", "gbm_full"]
    t = test[test.model.isin(ab)].pivot(index="model", columns="h", values="MAE").reindex(ab).dropna(how="all")
    if t.empty:
        return
    rel = (t / t.loc["gbm_base"] - 1) * 100
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(rel)
    for j, (m, r) in enumerate(rel.iterrows()):
        ax.bar(np.arange(len(r)) + j * w, r.values, w, label=nm(m))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(rel.columns)) + 0.4 - w / 2, [f"h = {h}" for h in rel.columns])
    ax.set_ylabel("изменение MAE к GBM база, %")
    ax.set_title("Абляция групп признаков GBM (test). Меньше = лучше")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    save(fig, "fig_ablation.png")


def fig_map():
    import geopandas as gpd
    p = ROOT / "results" / "forecast_2025.csv"
    if not p.exists():
        return
    f = pd.read_csv(p)
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    base = pd.DataFrame({"territory_id": meta.territory_id, "category": meta.category, "y": Y[:, -12:].sum(1)})
    g = f.groupby(["territory_id", "category"]).forecast.sum().reset_index().merge(base, on=["territory_id", "category"])
    g = g[g.category == "Все категории"]
    g["growth"] = (g.forecast / g.y - 1) * 100
    poly = gpd.read_file(ROOT / cfg["data"]["polygons"])
    poly = poly.sort_values("year_to").drop_duplicates("territory_id", keep="last")
    poly["territory_id"] = pd.to_numeric(poly["territory_id"], errors="coerce")
    poly = poly.dropna(subset=["territory_id"]).astype({"territory_id": int})
    poly = poly.merge(g, on="territory_id", how="inner").to_crs("EPSG:3576")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    poly.plot(column="growth", cmap="RdYlGn", vmin=0, vmax=25, ax=ax, linewidth=0.05, edgecolor="white",
              legend=True, legend_kwds={"label": "рост 2025 к 2024, %", "shrink": 0.6})
    ax.set_axis_off()
    ax.set_title("Прогноз: рост безналичных расходов на жителя в 2025 г. к 2024 г., «Все категории»")
    save(fig, "fig_map_2025.png")


def main():
    cfg = load_config()
    meta, Y, months = get_panel(cfg)
    test = pd.read_csv(MET / "metrics_test.csv")
    fmlocal = test[test.model.str.startswith("dec_fmlocal_")].groupby("model").MAE.mean().sort_values()
    best_fml = list(fmlocal.index[:1])
    fmlevel = test[test.model.str.startswith("fm_")].groupby("model").MAE.mean().sort_values()
    best_fm = list(fmlevel.index[:1])
    key = ["naive", "snaive_growth", "prophet", "autoets", "autotheta", "lgbm_v1"] + best_fm + \
        ["dec_persist"] + best_fml + ["gbm_full", "cat_full", "ensemble"]
    key = [k for k in key if k in set(test.model)]
    fig_national(meta, Y, months)
    fig_mae_h(test, key)
    fig_curve(key)
    fig_fm(test)
    fig_bootstrap(key)
    fig_cpd()
    fig_orsk(meta, Y, months)
    fig_examples(meta, Y, months)
    fig_ablation(test)
    fig_map()


if __name__ == "__main__":
    main()
