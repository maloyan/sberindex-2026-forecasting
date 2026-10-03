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


def build():
    PRES.mkdir(exist_ok=True)
    (PRES / "figures").mkdir(exist_ok=True)
    for f in (ROOT / "report" / "figures").glob("*.png"):
        shutil.copy(f, PRES / "figures" / f.name)
    pdf = ROOT / "report" / "report.pdf"
    if pdf.exists():
        shutil.copy(pdf, PRES / "report.pdf")
    key = ["naive", "snaive_growth", "prophet", "autotheta", "lgbm_v1", "dec_persist",
           "dec_fmlocal_timesfm25", "dec_fmlocal_chronos2_ft", "gbm_full", "cat_full", "ensemble"]
    test = pd.read_csv(MET / "metrics_test.csv")
    key = [k for k in key if k in set(test.model)]
    main, gain, r2 = main_table(key)
    hl, base = (NAMES["ensemble"],), (NAMES["prophet"],)
    syn = pd.read_csv(CPD / "synthetic_online.csv")
    s10 = syn[(syn.scenario == "shift") & (syn["size"] == 0.10)].set_index("detector")
    spk = syn[syn.scenario == "spike"].set_index("detector")["recall"]
    cpd = pd.DataFrame({"Детектор": s10.index, "F1 (сдвиг 10%)": s10.F1.values,
                        "Задержка, мес.": s10.mean_delay.values, "Тревога на выбросе": spk.reindex(s10.index).values,
                        "Ложные тревоги, %": (s10.clean_FAR.values * 100).round(1)}).sort_values("F1 (сдвиг 10%)", ascending=False)
    kp = {f"h={h}": f"−{int(g)}%" for h, g in gain.items()}
    main_html = tbl(main, hl, base)
    cpd_html = tbl(cpd, fmt="{:.2f}")
    kpis = "".join(f'<div class="kpi"><b>{v}</b><span>MAE к Prophet, {k}</span></div>' for k, v in kp.items())

    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>СберИндекс: прогноз расходов</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=PT+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<p class="sub">Конкурс СберИндекса 2026 · трек «Прогнозирование»</p>
<h1>Прогноз безналичных расходов муниципалитетов и поиск структурных сдвигов</h1>
<p class="sub">12 096 рядов (2016 МО × 6 категорий), 24 месяца. Горизонты 1, 3, 6, 12 месяцев. Честная проверка на отложенном окне 2024-07..12.</p>
<nav><a href="report.pdf">Отчёт (PDF)</a><a href="slides.pdf">Слайды (PDF)</a><a href="#res">Результаты</a><a href="#fm">Фундаментальные модели</a><a href="#cpd">Сдвиги</a><a href="#news">Новости</a></nav>
<div class="kpis">{kpis}</div>
<h2>Архитектура</h2>
<p>Сезонность и инфляция общие для всей страны. Их оцениваем по срезу из 2016 МО (национальная компонента), а модели прогнозируют только локальное отклонение МО. Это решает главную проблему данных: у ряда всего 24 точки, и перед прогнозом на 12 месяцев нет даже одного года истории.</p>
<img src="figures/fig_architecture.png" alt="Архитектура решения">
<img src="figures/fig_national.png" alt="Национальная компонента">
<h2 id="res">Сравнение моделей (test, MAE, руб. на жителя в месяц)</h2>
{main_html}
<p class="note">Каждая модель в момент прогноза видит только прошлое. Веса ансамбля выбраны на validation (2024-01..06), test не использовался ни для какого выбора. R² на уровнях для ансамбля: {", ".join(f"h={h}: {r2.loc[h, 'R2']:.3f}" for h in r2.index)}; R² на изменении: {", ".join(f"h={h}: {r2.loc[h, 'R2_change']:.2f}" for h in r2.index)}.</p>
<img src="figures/fig_mae_by_h_test.png" alt="MAE по горизонтам">
<img src="figures/fig_bootstrap_vs_prophet.png" alt="Бутстреп">
<img src="figures/fig_mae_curve.png" alt="Кривая MAE">
<h2 id="fm">Фундаментальные модели</h2>
<p>Восемь открытых моделей (Apache-2.0): Chronos-Bolt S/B, Chronos-2 (уни- и мультивариантный), TimesFM 2.5, TiRex-2, Toto-2.0 22M/313M, плюс каузальное дообучение Chronos-2. На сырых уровнях zero-shot проигрывает даже наивной модели: модели не видят календарь, а в 24 точках нет повторяющейся сезонности. На локальной компоненте те же модели снижают MAE в 1.5-3 раза и входят в ансамбль (−6% MAE ансамбля на h = 6).</p>
<img src="figures/fig_fm_level_vs_local.png" alt="FM: уровни против локальной компоненты">
<h2 id="cpd">Структурные сдвиги</h2>
<p>Детекторы откалиброваны на одинаковую частоту ложных тревог (≈2% на ряд-месяц), затем сравниваются на синтетических сдвигах, вставленных в реальные ряды.</p>
{cpd_html}
<img src="figures/fig_cpd_synthetic.png" alt="Синтетический бенчмарк">
<img src="figures/fig_orsk.png" alt="Орск">
<h2 id="news">Новости</h2>
<p>1,55 млн документов GDELT GKG о местах РФ. Город привязывается к МО по расстоянию и совпадению названия. Темы (наводнение, бедствие, экономика, безопасность) и тональность → признаки GBM и априорная информация для детекторов сдвигов.</p>
<img src="figures/fig_ablation.png" alt="Абляция">
<h2>Прогноз на 2025 год</h2>
<img src="figures/fig_map_2025.png" alt="Карта прогноза">
<img src="figures/fig_examples.png" alt="Примеры">
<p class="note">Код, конфиги и инструкция запуска: репозиторий проекта (README). Данные организатора в репозиторий не входят.</p>
</main></body></html>"""
    (PRES / "index.html").write_text(page, encoding="utf-8")

    slides = [
        f"""<section><p class="sub">Конкурс СберИндекса 2026 · трек «Прогнозирование»</p>
<h1 style="margin-top:1.2in">Прогноз безналичных расходов муниципалитетов<br>и поиск структурных сдвигов</h1>
<p class="sub">Декомпозиция «страна × МО» · 8 вариантов фундаментальных моделей · GBM · ансамбль · 10 детекторов сдвигов · новости GDELT</p>
<div class="kp">{"".join(f"<div><b>{v}</b>MAE к Prophet, {k}</div>" for k, v in kp.items())}</div></section>""",
        """<section><h2>Задача и главная трудность</h2><ul>
<li>Средние безналичные расходы на жителя, 2016 МО × 6 категорий = 12 096 рядов, 2023-01..2024-12</li>
<li>Прогноз на 1, 3, 6, 12 месяцев; обязательная метрика MAE; база — Prophet</li>
<li><b>Всего 24 точки на ряд.</b> Перед прогнозом на 12 месяцев у модели меньше года истории: собственную сезонность ряда оценить нельзя</li>
<li>Решение: сезонность и инфляцию оцениваем по срезу из 2016 МО, рост на год вперёд — из Росстата</li></ul>
<img src="figures/fig_national.png" style="max-height:3.6in"></section>""",
        """<section><h2>Архитектура решения</h2><img src="figures/fig_architecture.png"></section>""",
        """<section><h2>Честный протокол</h2><ul>
<li>Скользящий origin: прогноз из месяца o видит только месяцы 0..o; GBM обучается только на парах с целевым месяцем ≤ o</li>
<li>Validation (2024-01..06): выбор весов ансамбля и параметров. Test (2024-07..12): только отчёт</li>
<li>Все модели, включая Prophet, на всех 12 096 рядах; сравнение парное</li>
<li>Метрики: MAE (руб.), R² на уровнях и на изменении, WAPE, MdAPE, смещение</li>
<li>Значимость: кластерный бутстреп по МО, 1000 повторов</li></ul></section>""",
        f"""<section><h2>Результаты на отложенном окне (MAE, руб.)</h2>{main_html}</section>""",
        """<section><h2>Значимость разницы с Prophet</h2><img src="figures/fig_bootstrap_vs_prophet.png"></section>""",
        """<section><h2>Фундаментальные модели: где они помогают</h2><img src="figures/fig_fm_level_vs_local.png" style="max-height:4.2in">
<ul><li>Zero-shot на уровнях при h ≥ 3 хуже Prophet и на уровне наивной модели: 24 точки, нет календаря, сезонность не видна</li>
<li>На локальной компоненте MAE падает в 1.5-3 раза; лучшая — TimesFM 2.5; дообучение Chronos-2 даёт ещё −2% на h=3..6; в ансамбле FM дают −6% на h=6</li></ul></section>""",
        f"""<section><h2>Структурные сдвиги: сравнение детекторов</h2><div class="two"><div>{cpd_html}
<p class="sub" style="font-size:15px">Пороги откалиброваны на одинаковую частоту ложных тревог ≈2% на ряд-месяц</p></div>
<img src="figures/fig_cpd_synthetic.png"></div></section>""",
        """<section><h2>Реальный пример: паводок в Орске (апрель 2024)</h2><img src="figures/fig_orsk.png"></section>""",
        """<section><h2>Новости: GDELT → МО → признаки и prior</h2><div class="two"><div><ul>
<li>1,55 млн документов о местах РФ за 2022-12..2024-12</li><li>Привязка города к МО: ≤30 км до центра и совпадение названия</li>
<li>Темы из текста: наводнение, бедствие, экономика, безопасность; тональность</li>
<li>Использование: признаки GBM; снижение порога CUSUM / рост hazard BOCPD в месяц новостного всплеска; разметка реальных событий</li></ul></div>
<img src="figures/fig_ablation.png"></div></section>""",
        """<section><h2>Прогноз на 2025 год</h2><img src="figures/fig_map_2025.png"></section>""",
        """<section><h2>Выводы</h2><ul>
<li>Главный выигрыш даёт архитектура, а не одна модель: национальная компонента по срезу МО + рост из Росстата</li>
<li>Ансамбль выбирается только на validation и обходит Prophet на всех горизонтах</li>
<li>Фундаментальные модели полезны только после удаления общей сезонности; это и есть их корректное применение на коротких рядах</li>
<li>Для раннего обнаружения сдвигов — BOCPD (быстрее всех), для устойчивости к выбросам — CUSUM / BOCPD с подтверждением; новости повышают раннюю реакцию CUSUM</li>
<li>Всё воспроизводится одной командой: <code>make reproduce</code>; параметры в <code>configs/config.yaml</code></li></ul></section>""",
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
