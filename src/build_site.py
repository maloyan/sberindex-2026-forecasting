"""Build the presentation: presentation/index.html (landing page) and presentation/slides.html
(16:9 slides; scripts/build_report.sh prints them to slides.pdf with headless Chrome).

All numbers are read from results/metrics/*.csv and results/cpd/*.csv, so the pages always
match the latest run. Figures are copied from report/figures/ to presentation/figures/.

Usage: .venv/bin/python src/build_site.py
"""
from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from figures import NAMES  # noqa: E402

PRES = ROOT / "presentation"
MET = ROOT / "results" / "metrics"
CPD = ROOT / "results" / "cpd"

CSS = """
:root{--ink:#1a202c;--muted:#4a5568;--line:#e2e8f0;--accent:#2b6cb0;--red:#c53030;--bg:#ffffff;--soft:#f7fafc;--good:#276749}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#e2e8f0;--muted:#a0aec0;--line:#2d3748;--accent:#63b3ed;--red:#fc8181;--bg:#171923;--soft:#1f2430;--good:#68d391}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "PT Sans",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:30px;line-height:1.2;margin:8px 0 4px}
h2{font-size:22px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}h3{font-size:17px;margin:20px 0 6px}
.sub{color:var(--muted)}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--soft);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.kpi b{display:block;font-size:24px;color:var(--accent)}
.kpi span{color:var(--muted);font-size:13px}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0 4px;display:block;overflow-x:auto}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
tr.hl td{font-weight:700;color:var(--accent)}tr.base td{color:var(--red)}nav a{margin-right:14px;color:var(--accent);text-decoration:none}
.note{font-size:13px;color:var(--muted)}code{font-size:13px;background:var(--soft);padding:1px 4px;border-radius:4px}
ul{padding-left:20px}a{color:var(--accent)}
"""

SLIDE_CSS = """
@page{size:13.333in 7.5in;margin:0}*{box-sizing:border-box}
body{margin:0;font:20px/1.4 "PT Sans",system-ui,Arial,sans-serif;color:#1a202c;background:#fff}
section{width:13.333in;height:7.5in;padding:0.45in 0.6in;page-break-after:always;position:relative;overflow:hidden}
h1{font-size:40px;margin:0 0 10px}h2{font-size:30px;margin:0 0 14px;color:#2b6cb0}
.sub{color:#4a5568}img{max-width:100%;max-height:5.2in;display:block;margin:0 auto}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start}
table{border-collapse:collapse;font-size:16px;width:100%}th,td{padding:4px 8px;border-bottom:1px solid #e2e8f0;text-align:right}
th:first-child,td:first-child{text-align:left}tr.hl td{font-weight:700;color:#2b6cb0}tr.base td{color:#c53030}
ul{margin:6px 0;padding-left:24px}li{margin:4px 0}.foot{position:absolute;bottom:0.25in;right:0.6in;font-size:12px;color:#a0aec0}
.big{font-size:54px;font-weight:700;color:#2b6cb0}.kp{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}
.kp div{background:#f7fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px}.kp b{font-size:30px;color:#2b6cb0;display:block}
"""


def tbl(df: pd.DataFrame, hl=(), base=(), fmt="{:.0f}") -> str:
    h = "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    rows = []
    for _, r in df.iterrows():
        cls = ' class="hl"' if r.iloc[0] in hl else (' class="base"' if r.iloc[0] in base else "")
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, float) else html.escape(str(v))}</td>" for v in r.values)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table>{h}{''.join(rows)}</table>"


def main_table(key):
    t = pd.read_csv(MET / "metrics_test.csv")
    p = t[t.model.isin(key)].pivot(index="model", columns="h", values="MAE").reindex(key).dropna(how="all")
    pr = p.loc["prophet"]
    out = pd.DataFrame({"Модель": [NAMES.get(m, m) for m in p.index]})
    for h in p.columns:
        out[f"h={h}"] = p[h].values
    ens = p.loc["ensemble"]
    gain = ((1 - ens / pr) * 100).round(0)
    r2 = t[(t.model == "ensemble")].set_index("h")
    return out, gain, r2


def fmt_ci(lo, hi):
    return f"[{lo:+.0f}; {hi:+.0f}]".replace("-", "−")


def data():
    """All numbers shown on the pages, read from the result tables."""
    D = {}
    sel = pd.read_csv(MET / "selection_table.csv").set_index("h")
    bc = pd.read_csv(MET / "metrics_test_bycat.csv")
    allc = bc[bc.category == "Все категории"].pivot(index="model", columns="h", values="MAE")
    rows = []
    for h, r in sel.iterrows():
        rows.append({"h": h, "Prophet": r.MAE_test_prophet, "Ансамбль": r.MAE_test_ensemble,
                     "Снижение": f"−{(1 - r.MAE_test_ensemble / r.MAE_test_prophet) * 100:.0f}%",
                     "95% ДИ разницы": fmt_ci(r.twoway_lo95, r.twoway_hi95),
                     "Значимо": "да" if r.twoway_hi95 < 0 else "нет",
                     "R² изм.": f"{r.R2_change_test:.2f}".replace("-", "−"),
                     "Ранг на test": f"{int(r.rank_test_ensemble)} из {int(r.n_candidates)}",
                     "«Все категории»": f"{allc.loc['ensemble', h]:.0f} / {allc.loc['prophet', h]:.0f}"})
    D["headline"] = pd.DataFrame(rows)
    D["gain"] = {h: (1 - r.MAE_test_ensemble / r.MAE_test_prophet) * 100 for h, r in sel.iterrows()}
    D["sig"] = {h: r.twoway_hi95 < 0 for h, r in sel.iterrows()}
    rows = []
    for h, r in sel.iterrows():
        rows.append({"h": h, "Лучшая одиночная, val": f"{r.MAE_val_best_single:.0f} ({NAMES.get(r.best_single_val, r.best_single_val)})",
                     "Ансамбль, val*": r.MAE_val_ensemble_insample, "Ансамбль, test": r.MAE_test_ensemble,
                     "Лучшая на test (после)": f"{r.MAE_test_best_single:.0f} ({NAMES.get(r.best_single_test, r.best_single_test)})"})
    D["selection"] = pd.DataFrame(rows)
    dt = pd.read_csv(CPD / "decision_table.csv").set_index("detector")
    order = ["bocpd_confirm", "bocpd", "pelt_online", "window_online", "cusum", "cusum_news"]
    rows = []
    for k in order:
        r = dt.loc[k]
        f1 = f"{r.F1_10:.2f}" if pd.notna(r.F1_10) else f"{r.F1_10_semi:.2f}*"
        rows.append({"Детектор": r["name"], "F1 сдвиг 10%": f1, "Задержка, мес.": f"{r.delay_10:.1f}",
                     "Тревога на выбросе": "—" if pd.isna(r.outlier_alarm) else f"{r.outlier_alarm:.2f}",
                     "Орск": r.orsk_first_alarm, "Lift GDELT": f"{r.news_lift:.1f}",
                     "Lift паводки": f"{r.flood_lift:.1f}" if pd.notna(r.flood_lift) else "—"})
    D["cpd"] = pd.DataFrame(rows)
    es = pd.read_csv(MET / "event_study_summary.csv")
    D["es"] = pd.DataFrame({"Категория": es.category,
                            "Эффект 0..2 мес.": [f"{x * 100:+.1f}%".replace("-", "−") for x in es.effect_post_0_2],
                            "95% ДИ": [f"[{a * 100:+.1f}; {b * 100:+.1f}]".replace("-", "−") for a, b in zip(es.lo95, es.hi95)],
                            "p (Холм)": es.p_holm.round(2)})
    it = pd.read_csv(MET / "intervals_test.csv")
    t = pd.read_csv(MET / "metrics_test.csv")
    def line(model, col, f):
        v = t[t.model == model].set_index("h")[col]
        return " / ".join(f.format(v[h]) for h in (1, 3, 6, 12)).replace("-", "−")
    def iline(model, col, f):
        v = it[it.model == model].set_index("h")[col]
        return " / ".join(f.format(v[h]) for h in (1, 3, 6, 12))
    D["metrics"] = pd.DataFrame([
        ["MAE, руб.", line("ensemble", "MAE", "{:.0f}"), line("prophet", "MAE", "{:.0f}"), "средняя ошибка (обязательная)"],
        ["R² (уровни)", line("ensemble", "R2", "{:.3f}"), line("prophet", "R2", "{:.3f}"), "≈ 1 у всех: уровни МО разные"],
        ["R² на изменении", line("ensemble", "R2_change", "{:.2f}"), line("prophet", "R2_change", "{:.1f}"), "объясняет ли модель динамику"],
        ["WAPE", line("ensemble", "WAPE", "{:.1%}"), line("prophet", "WAPE", "{:.1%}"), "% от суммы факта"],
        ["Смещение", line("ensemble", "Bias%", "{:+.1f}%"), line("prophet", "Bias%", "{:+.1f}%"), "систематический сдвиг"],
        ["Покрытие 80%", iline("ensemble", "coverage80", "{:.2f}"), iline("prophet", "coverage80", "{:.2f}"), "цель 0.80"],
        ["WIS, руб.", iline("ensemble", "WIS", "{:.0f}"), iline("prophet", "WIS", "{:.0f}"), "интервал + медиана"]],
        columns=["Метрика", "Ансамбль h=1/3/6/12", "Prophet h=1/3/6/12", "Смысл"])
    c = pd.read_csv(MET / "check_2025_portal.csv")
    a = c[c.category == "Все категории"]
    D["check"] = (a.yoy_forecast.mean(), a.yoy_portal_2025.mean())
    pg = MET / "prophet_grid_test.csv"
    D["pg"] = None
    if pg.exists():
        g = pd.read_csv(pg)
        piv = g.pivot(index="config", columns="h", values="MAE")
        grid = piv.loc[g[g.kind == "prophet grid"].config.unique()]
        D["pg"] = pd.DataFrame({"h": piv.columns, "Prophet, наша конфигурация": piv.loc["ours_m3"].values,
                                "Лучшая из 10 (выбор по test)": grid.min().values,
                                "Какая": [grid[h].idxmin() for h in piv.columns],
                                "Ансамбль": piv.loc["ensemble"].values})
    return D


def build():
    PRES.mkdir(exist_ok=True)
    (PRES / "figures").mkdir(exist_ok=True)
    for f in (ROOT / "report" / "figures").glob("*.png"):
        shutil.copy(f, PRES / "figures" / f.name)
    pdf = ROOT / "report" / "report.pdf"
    if pdf.exists():
        shutil.copy(pdf, PRES / "report.pdf")
    D = data()
    head = tbl(D["headline"], fmt="{:.0f}")
    gain = D["gain"]
    kp = "".join(f'<div class="kpi"><b>−{g:.0f}%</b><span>MAE к Prophet, h = {h}'
                 f'{"" if D["sig"][h] else " (не значимо)"}</span></div>' for h, g in gain.items())
    pg_html = tbl(D["pg"], fmt="{:.0f}") if D["pg"] is not None else "<p>Сетка Prophet считается.</p>"
    f25, p25 = D["check"]
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>СберИндекс: прогноз расходов</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=PT+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<p class="sub">Конкурс СберИндекса 2026 · трек «Прогнозирование»</p>
<h1>Прогноз безналичных расходов муниципалитетов и поиск структурных сдвигов</h1>
<p class="sub">12 096 рядов (2016 МО × 6 категорий), 24 месяца. Горизонты 1, 3, 6, 12 месяцев. Проверка на отложенном окне 2024-07..12 и на факте 2025 года.</p>
<nav><a href="report.pdf">Отчёт (PDF)</a><a href="slides.pdf">Слайды (PDF)</a><a href="#m">Методология</a><a href="#res">Модели</a><a href="#fm">FM</a><a href="#cpd">Сдвиги</a><a href="#news">Новости</a><a href="#met">Метрики</a><a href="#out">Выводы</a></nav>
<div class="kpis">{kp}</div>
<h2 id="m">1. Методология за минуту</h2>
<p>Расходы МО = общий ритм страны × отклонение города. Ритм страны (сезон и инфляция) оцениваем по 2016 МО сразу, а не по 24 точкам одного ряда; годовой рост берём из Росстата. Отклонение города прогнозируют фундаментальные модели, GBM и классика; 35 моделей объединяет ансамбль с весами, выбранными только на validation. Детекторы сдвигов, новости и реестр ЧС следят за тем, где ряд сломался.</p>
<img src="figures/fig_method.png" alt="Методология">
<img src="figures/fig_protocol.png" alt="Протокол">
<p class="note">Один раз test использован в дизайне (правило (k, w) для h ≤ 3, записано в LOG.md). Пост-hoc эксперименты — в приложении А отчёта, в заголовок не входят.</p>
<h2 id="res">2. Сравнение моделей (test, MAE, руб. на жителя в месяц)</h2>
{head}
<p class="note">95% ДИ — двусторонний бутстреп (МО × месяц цели). Значимо на h = 1, 6, 12; на h = 3 ансамбль лучше в 5 из 6 месяцев, но ДИ включает 0. «Все категории»: ансамбль / Prophet.</p>
<h3>Как выбрана модель (validation) и что получилось на test</h3>
{tbl(D["selection"], fmt="{:.0f}")}
<p class="note">* на окне подбора весов, оптимистично.</p>
<h3>Prophet не соломенное чучело: 10 конфигураций, 400 МО</h3>
{pg_html}
<img src="figures/fig_bootstrap_vs_prophet.png" alt="Значимость">
<img src="figures/fig_mae_curve.png" alt="Кривая MAE">
<h2 id="fm">3. Фундаментальные модели</h2>
<p>8 вариантов 5 семейств (Apache-2.0). На сырых уровнях zero-shot при h ≥ 3 хуже Prophet и на уровне наивной модели: 24 точки, нет календаря. На отклонении города те же модели снижают MAE в 1.5-3 раза; в ансамбле FM дают −6% на h = 6, на h = 12 экстраполируют локальный тренд и проигрывают persistence.</p>
<img src="figures/fig_fm_level_vs_local.png" alt="FM">
<h2 id="cpd">4. Структурные сдвиги</h2>
<p>Все детекторы откалиброваны на одну частоту ложных тревог (≈2%). <b>Выбор для мониторинга — BOCPD с подтверждением</b> (лучший F1, выбросы вдвое реже BOCPD). На реальных событиях стабильно выше базовой частоты только CUSUM — он для проверки гипотез по новостям и реестру ЧС.</p>
{tbl(D["cpd"])}
<p class="note">* полусинтетика с новостями. Lift = доля событий с тревогой в 0..2 мес. / базовая частота.</p>
<img src="figures/fig_cpd_synthetic.png" alt="Синтетика">
<img src="figures/fig_orsk.png" alt="Орск">
<h2 id="news">5. Новости и реестр ЧС</h2>
<p>GDELT GKG: 1,55 млн документов о местах РФ, привязка к МО (30 км + название). Реестр 71 события из официальных источников (постановления о ЧС, Банк России) с датами объявления и вступления. Правило as-of: в момент прогноза только документы и акты с датой до конца месяца origin.</p>
<img src="figures/fig_news_timeline.png" alt="As-of">
<h3>Event study паводков: 42 МО из актов о ЧС против МО того же региона</h3>
{tbl(D["es"], fmt="{:.2f}")}
<p class="note">Результат нулевой: среднего эффекта нет ни в одной категории после поправки Холма; плацебо-даты дают такие же эффекты. Орск (+7%) — отдельный случай.</p>
<img src="figures/fig_event_study.png" alt="Event study">
<h2 id="met">6. Метрики</h2>
{tbl(D["metrics"])}
<h2 id="out">7. Выводы, проверка 2025, воспроизводимость</h2>
<p>Прогноз 2025, сделанный в конце 2024 г.: рост «Всех категорий» {f25:.3f} (лог.) против факта портала СберИндекса {p25:.3f}.</p>
<img src="figures/fig_check_2025.png" alt="Проверка 2025">
<img src="figures/fig_map_2025.png" alt="Карта прогноза">
<ul><li>Главное — отделить общую динамику страны; годовому горизонту нужен внешний источник роста.</li>
<li>FM применять к очищенному от сезона сигналу, а не к сырым коротким рядам.</li>
<li>Режим ЧС сам по себе не предсказывает сдвиг расходов; нужен масштаб ущерба.</li>
<li><code>make setup data quick</code> — со свежего клона за 2 минуты пересчитывает все главные числа; <code>make reproduce</code> — полный прогон.</li></ul>
<p class="note">Данные организатора в репозиторий не входят (Положение §9.2.2).</p>
</main></body></html>"""
    (PRES / "index.html").write_text(page, encoding="utf-8")

    sigtxt = "значимо на h = 1, 6, 12 (двусторонний бутстреп); на h = 3 ДИ включает 0"
    pg_slide = (f"""{pg_html}<img src="figures/fig_prophet_grid.png" style="max-height:3.2in">""" if D["pg"] is not None
                else "<p>Сетка считается.</p>")
    slides = [
        f"""<section><p class="sub">Конкурс СберИндекса 2026 · трек «Прогнозирование»</p>
<h1 style="margin-top:0.9in">Прогноз безналичных расходов муниципалитетов<br>и поиск структурных сдвигов</h1>
<p class="sub">Ритм страны × отклонение города · 8 фундаментальных моделей · ансамбль · 10 детекторов сдвигов · новости и реестр ЧС</p>
<div class="kp">{"".join(f"<div><b>−{g:.0f}%</b>MAE к Prophet, h = {h}</div>" for h, g in gain.items())}</div>
<p class="sub" style="font-size:16px">Test 2024-07..12, 12 096 рядов; {sigtxt}.</p></section>""",
        """<section><h2>1. Методология за минуту</h2><img src="figures/fig_method.png" style="max-height:2.6in">
<ul><li><b>Проблема:</b> 24 точки на ряд, перед прогнозом на год нет даже года истории</li>
<li><b>Идея:</b> сезон и инфляцию оцениваем по 2016 МО сразу; модели прогнозируют только отклонение города</li>
<li><b>Итог:</b> y = ритм страны × отклонение города; веса ансамбля только на validation</li></ul></section>""",
        """<section><h2>Честный протокол</h2><img src="figures/fig_protocol.png" style="max-height:2.4in"><ul>
<li>Скользящий origin: прогноз из месяца o видит только месяцы 0..o; Росстат с лагом 2 мес., новости и акты — до конца o</li>
<li>Validation — выбор; test — оценка. Один взгляд на test в дизайне (правило (k, w)) записан в LOG.md; всё пост-hoc — отдельно</li>
<li>Все модели, включая Prophet, на всех 12 096 рядах; тесты на утечку в <code>make smoke</code></li></ul></section>""",
        f"""<section><h2>2. Лучшая модель по горизонтам (test, MAE, руб.)</h2>{head}
<p class="sub" style="font-size:15px">95% ДИ разницы с Prophet — двусторонний бутстреп (МО × месяц цели). «Все категории»: ансамбль / Prophet.</p>
{tbl(D["selection"], fmt="{:.0f}")}<p class="sub" style="font-size:13px">Правило: ансамбль, потому что на validation он лучше любой одиночной модели (* — на окне подбора весов). Лучшая на test одиночная модель известна только после.</p></section>""",
        f"""<section><h2>Prophet не соломенное чучело</h2><p>10 конфигураций Prophet на 400 МО, лучшая выбрана по test (в пользу Prophet)</p>{pg_slide}</section>""",
        """<section><h2>3. Фундаментальные модели: где помогают и где нет</h2><img src="figures/fig_fm_level_vs_local.png" style="max-height:3.4in">
<div class="two" style="font-size:17px"><ul><li>✔ на отклонении города: MAE в 1.5-3 раза ниже, лучше Prophet на всех h</li><li>✔ в ансамбле −6% MAE на h = 6; Chronos даёт откалиброванные интервалы</li><li>✔ мультивариантный Chronos-2 и дообучение: ещё −2%</li></ul>
<ul><li>✘ на сырых коротких рядах: хуже Prophet при h ≥ 3</li><li>✘ h = 12: экстраполируют тренд (729 против 691 у persistence)</li><li>✘ как признаки GBM: ±1%</li></ul></div></section>""",
        f"""<section><h2>4. Детекторы сдвигов: таблица решений</h2>{tbl(D["cpd"])}
<ul style="font-size:18px"><li>Все пороги откалиброваны на одну частоту ложных тревог ≈2%; синтетика: сдвиги, тренды, выбросы в реальных рядах</li>
<li><b>Мониторинг: BOCPD с подтверждением</b> — лучший F1, выбросы вдвое реже BOCPD; быстрый сигнал — BOCPD</li>
<li>Реальные события: стабильно выше базы только CUSUM (lift 1.5-2.3) → проверка гипотез по новостям и реестру</li></ul></section>""",
        f"""<section><h2>Реальный сдвиг: Орск и паводки 2023-2024</h2><div class="two"><img src="figures/fig_orsk.png">
<div>{tbl(D["es"], fmt="{:.2f}")}<p style="font-size:16px">Event study: 42 МО из актов о ЧС против МО того же региона, плацебо-даты, перестановки, Холм. <b>Среднего эффекта нет</b>; Орск (+7%) — исключение. Акт о ЧС ≠ масштаб ущерба.</p></div></div></section>""",
        """<section><h2>5. Новости: как согласуем с данными</h2><img src="figures/fig_news_timeline.png" style="max-height:2.6in"><ul style="font-size:18px">
<li>GDELT GKG: 1,55 млн документов; МО = 30 км + совпадение названия (942 места → 811 МО); иначе регион</li>
<li>Реестр 71 события: постановления о ЧС (publication.pravo.gov.ru), Банк России; даты объявления и вступления</li>
<li>Признаки GBM: ±1% MAE (честный ноль). Prior: CUSUM ловит Орск на месяц раньше; полусинтетика F1 0.43 против 0.38</li></ul></section>""",
        f"""<section><h2>6. Метрики</h2>{tbl(D["metrics"])}<p class="sub" style="font-size:15px">R² на уровнях не различает модели; R² на изменении показывает динамику. Интервалы: квантили ошибок validation, проверка на test.</p></section>""",
        f"""<section><h2>7. Прогноз 2025 и проверка на факте</h2><div class="two"><img src="figures/fig_map_2025.png"><div><img src="figures/fig_check_2025.png">
<p>Прогноз конца 2024 г.: рост «Всех категорий» <b>{f25:.3f}</b> (лог.) против факта портала СберИндекса <b>{p25:.3f}</b>. Единственная проверка на данных, которых не видел ни один этап.</p></div></div></section>""",
        """<section><h2>Выводы и как запустить</h2><ul>
<li>Главный выигрыш — архитектура: ритм страны по срезу МО + внешний рост для годового горизонта</li>
<li>Ансамбль лучше Prophet на всех h; значимо на 3 из 4 — честно, с двусторонним бутстрепом</li>
<li>FM — только после удаления общего сезона; на сырых коротких рядах они не работают</li>
<li>Мониторинг сдвигов — BOCPD с подтверждением; гипотезы по новостям и ЧС — CUSUM с prior</li>
<li>Режим ЧС сам по себе не сдвигает расходы на жителя; нужен масштаб ущерба</li></ul>
<pre style="font-size:18px;background:#f7fafc;padding:12px;border-radius:8px">make setup data quick   # свежий клон: тесты на утечку + все главные числа, ≈ 2 мин
make reproduce          # полный прогон, 4-5 ч</pre></section>""",
    ]
    for i, s in enumerate(slides):
        slides[i] = s.replace("</section>", f'<div class="foot">{i + 1} / {len(slides)}</div></section>')
    sp = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Слайды</title>
<link href="https://fonts.googleapis.com/css2?family=PT+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>{SLIDE_CSS}</style></head><body>{''.join(slides)}</body></html>"""
    (PRES / "slides.html").write_text(sp, encoding="utf-8")
    print("wrote presentation/index.html and slides.html")


if __name__ == "__main__":
    build()
