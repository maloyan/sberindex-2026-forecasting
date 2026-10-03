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
        sig = s.twoway_hi95 < 0
        ax.errorbar(s.dMAE, y, xerr=[s.dMAE - s.twoway_lo95, s.twoway_hi95 - s.dMAE], fmt="none",
                    ecolor=C_GREY, capsize=2)
        ax.scatter(s.dMAE[sig], y[sig.to_numpy()], color=C_MAIN, s=18, zorder=3, label="ДИ ниже нуля")
        ax.scatter(s.dMAE[~sig], y[~sig.to_numpy()], facecolor="white", edgecolor=C_MAIN, s=18, zorder=3,
                   label="ДИ включает 0")
        ax.axvline(0, color=C_PROPHET, lw=1)
        ax.set_title(f"h = {h}"); ax.set_xlabel("MAE модели - MAE Prophet, руб.")
        ax.set_yticks(y, [nm(m) for m in s.index], fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Разница с Prophet, 95% ДИ двустороннего бутстрепа (МО x месяц цели), test", y=1.04)
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


def _box(ax, x, y, w, h, title, body, fc):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="#4a5568", lw=1))
    ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=11.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.26, body, ha="center", va="top", fontsize=9, linespacing=1.35)


def fig_method():
    """Methodology in one picture: 4 plain-language boxes, left to right, no crossing edges."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(0, 4.3); ax.set_ylim(0.25, 1.12); ax.axis("off")
    boxes = [
        ("1. Общий ритм страны", "сезон и инфляция:\nмедиана по 2016 МО\nв каждом месяце\n+ годовой рост из Росстата", "#fefcbf"),
        ("2. Отклонение города", "что остаётся после шага 1:\nмедленный ряд без сезона.\nЕго прогнозируют FM,\nGBM и классика", "#ebf4ff"),
        ("3. Сложить и усреднить", "прогноз = ритм x отклонение.\n35 моделей, веса по\nгоризонту только на\nvalidation (Caruana)", "#f0fff4"),
        ("4. Сторож", "детекторы сдвигов на шаге 2\n+ новости и реестр ЧС:\nгде ряд сломался\nи прогнозу нельзя верить", "#fff5f5"),
    ]
    for j, (t, b, c) in enumerate(boxes):
        _box(ax, 0.05 + j * 1.07, 0.3, 0.95, 0.8, t, b, c)
        if j < 3:
            ax.annotate("", xy=(0.05 + (j + 1) * 1.07, 0.7), xytext=(0.05 + j * 1.07 + 0.95, 0.7),
                        arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=1.5))
    ax.set_title("Методология за минуту:  расходы МО = общий ритм страны x отклонение города", fontsize=12)
    save(fig, "fig_method.png")


def fig_protocol(months):
    """Timeline of the honest protocol: history, validation, test, 2025 check."""
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(-0.5, 36.5); ax.set_ylim(0, 3); ax.axis("off")
    spans = [(0, 12, "#edf2f7", "2023: только история"), (12, 18, "#fefcbf", "validation\n2024-01..06\nвыбор весов"),
             (18, 24, "#c6f6d5", "test\n2024-07..12\nитоговые числа"), (24, 36, "#e9d8fd", "2025: проверка прогноза\nпо порталу СберИндекса")]
    for a, b, c, t in spans:
        ax.add_patch(plt.Rectangle((a - 0.5, 1.2), b - a, 1.0, fc=c, ec="#4a5568", lw=0.8))
        ax.text((a + b - 1) / 2, 1.7, t, ha="center", va="center", fontsize=9)
    for i in range(0, 37, 3):
        lab = f"{2023 + i // 12}-{i % 12 + 1:02d}"
        ax.text(i, 1.05, lab, ha="center", va="top", fontsize=7, color="#4a5568")
    o, h = 17, 3
    ax.annotate("", xy=(o + h, 2.55), xytext=(o, 2.55), arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", lw=1.6))
    ax.plot([o], [2.55], "o", color="#2b6cb0")
    ax.text(o - 0.3, 2.8, "origin o: модель видит месяцы 0..o", fontsize=8.5, color="#2b6cb0", ha="right")
    ax.text(o + h + 0.3, 2.8, "цель o + h", fontsize=8.5, color="#2b6cb0")
    ax.text(0, 0.35, "Скользящий origin: прогноз из каждого месяца o на h = 1..12. Новости, Росстат и портал берутся "
            "только с датой <= конца месяца o. Test использован один раз для дизайна (правило (k, w), LOG.md).",
            fontsize=8.5, color="#4a5568")
    save(fig, "fig_protocol.png")


def fig_news_timeline():
    """As-of alignment of news with spending: what is known at origin o."""
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    for k, lab in enumerate(["месяц o - 1", "месяц o", "месяц o + 1", "...", "месяц o + h"]):
        x = 0.3 + k * 2.3
        ax.add_patch(plt.Rectangle((x, 1.4), 2.1, 0.5, fc="#edf2f7" if k < 2 else "#ffffff", ec="#4a5568", lw=0.8))
        ax.text(x + 1.05, 1.65, lab, ha="center", va="center", fontsize=9)
    ax.axvline(0.3 + 2 * 2.3, ymin=0.25, ymax=0.85, color="#c53030", lw=2)
    ax.text(0.3 + 2 * 2.3 + 0.08, 2.75, "конец месяца o = момент прогноза", color="#c53030", fontsize=9)
    ax.text(0.3, 2.6, "GDELT: дата публикации (UTC) -> месяц.\nПризнаки origin o: документы с датой <= конца o", fontsize=8.5)
    ax.text(0.3, 0.95, "СберИндекс: месяц o известен после его конца\n(тот же момент, что и новости)", fontsize=8.5)
    ax.text(5.0, 0.95, "Детектор в месяце t: новостной всплеск в t или t - 1\nснижает порог CUSUM (x0.6), hazard BOCPD x5", fontsize=8.5)
    ax.text(9.6, 2.6, "цель: только факт\nдля оценки ошибки", fontsize=8.5, ha="center")
    ax.text(0.3, 0.25, "Геопривязка: точка GDELT -> МО, если центр МО в 30 км и совпадают первые 5 букв названия; "
            "иначе уровень региона (ADM1). Реестр ЧС: дата объявления режима <= конца o.", fontsize=8.5, color="#4a5568")
    save(fig, "fig_news_timeline.png")


def fig_check_2025():
    p = MET / "check_2025_portal.csv"
    if not p.exists():
        return
    c = pd.read_csv(p)
    a = c[c.category == "Все категории"]
    x = pd.to_datetime(a.month)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(x, a.yoy_portal_2025 * 100, color="black", marker="o", ms=3, label="факт: портал СберИндекса (вся РФ)")
    ax.plot(x, a.yoy_forecast * 100, color=C_MAIN, marker="o", ms=3, label="наш прогноз из 2024-12 (медиана по МО)")
    ax.set_ylabel("log(2025-м / 2024-м) x 100"); ax.set_ylim(0, 20)
    ax.set_title(f"Проверка прогноза 2025 на факте, «Все категории»: лог-прирост в среднем {a.yoy_forecast.mean():.3f} "
                 f"против {a.yoy_portal_2025.mean():.3f}", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "fig_check_2025.png")


def fig_prophet_grid():
    p = MET / "prophet_grid_test.csv"
    if not p.exists():
        return
    g = pd.read_csv(p)
    piv = g.pivot(index="config", columns="h", values="MAE")
    grid = piv.loc[g[g.kind == "prophet grid"].config.unique()]
    best = grid.min()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    H = list(piv.columns)
    xs = np.arange(len(H))
    for cfgname, r in grid.iterrows():
        ax.plot(xs, r.values, color=C_GREY, lw=0.8, alpha=0.8)
    ax.plot(xs, best.values, color=C_PROPHET, lw=2, marker="o", label="лучшая из 10 конфигураций Prophet (на каждом h)")
    if "ensemble" in piv.index:
        ax.plot(xs, piv.loc["ensemble"].values, color=C_MAIN, lw=2.4, marker="o", label="наш ансамбль v1")
    ax.set_xticks(xs, [f"h = {h}" for h in H]); ax.set_ylabel("MAE, руб. (400 МО, test)")
    ax.set_title("Prophet не соломенное чучело: 10 конфигураций (серые) против ансамбля", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig_prophet_grid.png")


def fig_event_study():
    p = MET / "event_study_flood.csv"
    if not p.exists():
        return
    e = pd.read_csv(p)
    e = e[e.spec == "main"]
    cats = list(e.category.unique())
    fig, axes = plt.subplots(1, len(cats), figsize=(2.1 * len(cats), 3.0), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cats):
        s = e[e.category == c].sort_values("rel_month")
        ax.errorbar(s.rel_month, s.att * 100, yerr=[(s.att - s.lo95) * 100, (s.hi95 - s.att) * 100],
                    fmt="o-", ms=3, color=C_MAIN, ecolor=C_GREY, capsize=2)
        ax.axhline(0, color="black", lw=0.7); ax.axvline(-0.5, color=C_PROPHET, ls=":", lw=1)
        ax.set_title(c, fontsize=8.5); ax.set_xlabel("месяц к паводку")
    np.atleast_1d(axes)[0].set_ylabel("эффект на d, % (95% ДИ)")
    fig.suptitle("Паводки с режимом ЧС (Приморье 2023, Оренбург, Курган, Тюмень, Омск 2024): 42 МО из актов против МО того же региона",
                 y=1.06, fontsize=10)
    save(fig, "fig_event_study.png")


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
    fig_method()
    fig_protocol(months)
    fig_news_timeline()
    fig_check_2025()
    fig_prophet_grid()
    fig_event_study()


if __name__ == "__main__":
    main()
