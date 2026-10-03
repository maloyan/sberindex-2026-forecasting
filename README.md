# СберИндекс 2026 · Прогнозирование: расходы муниципалитетов и структурные сдвиги

Решение для направления «Прогнозирование» конкурса СберИндекса 2026.
Прогноз средних безналичных расходов на жителя для 2016 МО × 6 категорий (12 096 рядов) на
1, 3, 6 и 12 месяцев и поиск точек структурных сдвигов.

- **Отчёт (методология, на русском):** [`report/report.pdf`](report/report.pdf) (исходник [`report/report.md`](report/report.md))
- **Презентация:** [`presentation/slides.pdf`](presentation/slides.pdf) и лендинг [`presentation/index.html`](presentation/index.html)
- **Журнал экспериментов:** [`LOG.md`](LOG.md)
- **Прогноз на 2025 год:** [`results/forecast_2025.csv`](results/forecast_2025.csv)

## Результат (отложенное окно 2024-07..12, MAE, руб. на жителя в месяц)

| Модель | h = 1 | h = 3 | h = 6 | h = 12 |
|---|---:|---:|---:|---:|
| Prophet (базовая модель организатора) | 563 | 574 | 652 | 1361 |
| Лучшая zero-shot FM на сырых уровнях (TimesFM 2.5) | 559 | 797 | 1283 | 1691 |
| TimesFM 2.5 на локальной компоненте | 296 | 522 | 493 | 729 |
| GBM (LightGBM, все признаки) | 296 | 532 | 509 | 724 |
| **Ансамбль** | **306** | **435** | **446** | **658** |
| Снижение MAE ансамбля к Prophet | −46% | −24% | −32% | −52% |

Все модели в момент прогноза видят только прошлое (скользящий origin). Веса ансамбля и
параметры выбраны на validation (2024-01..06); test (2024-07..12) не использовался ни для
какого выбора. Разница с Prophet значима (кластерный бутстреп по МО, 95% ДИ ниже нуля на всех h).
Дополнительно: R², R² на изменении, WAPE, MdAPE, смещение — в отчёте, раздел 6.2.

## Идея

```
log y[i,t] = n[c,t] + d[i,t]
```

- **n** — национальная компонента категории (медиана по 2016 МО): сезонность и инфляция общие для
  всей страны. Прогноз: сезонная наивная + годовой рост (собственный и по обороту розницы Росстата).
- **d** — локальная компонента МО. Её прогнозируют фундаментальные модели (Chronos-Bolt, Chronos-2,
  TimesFM 2.5, TiRex-2, Toto-2.0; zero-shot и дообучение), LightGBM/CatBoost, классика.
- **Ансамбль Caruana** по горизонтам. **Детекторы сдвигов** (BOCPD, CUSUM, PELT, BinSeg, Window,
  Kernel и др.) на d с калибровкой на равную частоту ложных тревог; **новости GDELT** как признаки и prior.

![Архитектура](report/figures/fig_architecture.png)

## Запуск

Нужно: macOS или Linux, [uv](https://docs.astral.sh/uv/), ≈ 40 ГБ трафика для новостей GDELT,
pandoc + XeLaTeX и Google Chrome только для сборки PDF.

```bash
make setup        # .venv на Python 3.13, точные версии из requirements.lock.txt
make data         # данные СберИндекса, справочник МО, Росстат -> data/raw/ (в git не входят)
make reproduce    # полный прогон: ≈ 4-5 ч на Apple M3 Pro (FM на MPS; на CUDA быстрее)
make quick        # только ансамбль, метрики, прогноз, рисунки и отчёт по готовому кэшу
```

Отдельные шаги: `make baselines prophet statsf decomp fm fm-local finetune decomp-fm cpd
break-aware news gbm ensemble evaluate analysis forecast figures report` (см. `Makefile`).
Устройство для FM: `make fm FM_DEVICE=cuda|mps|cpu`.

## Конфигурация

Все гиперпараметры — в [`configs/config.yaml`](configs/config.yaml): окна validation/test,
горизонты, параметры Prophet, LightGBM, CatBoost, statsforecast, список фундаментальных моделей,
детекторы сдвигов и их сетки, группы тем новостей, радиус геопривязки.

## Структура

```
configs/config.yaml        все параметры
src/common.py              панель, «кубы» прогнозов (origin x horizon), метрики
src/models/                baselines, Prophet, statsforecast, декомпозиция, GBM, LightGBM v1, учёт сдвигов
src/fm/                    адаптеры и прогон фундаментальных моделей, дообучение Chronos-2
src/cpd/                   детекторы сдвигов, бенчмарк, признаки для GBM
src/news/                  загрузка GDELT GKG, геопривязка к МО, признаки
src/exog/                  Росстат, производственный календарь
src/ensemble.py            отбор Caruana по горизонтам (validation)
src/evaluate.py            метрики validation/test, кластерный бутстреп
src/analysis.py            абляции, FM-таблица, вклад FM в ансамбль
src/forecast_final.py      прогноз 2025 + 80% интервалы
src/figures.py, src/build_site.py, scripts/build_report.sh   рисунки, презентация, PDF
results/metrics, results/cpd   все таблицы, на которые ссылается отчёт
data/external              производные открытые данные: Росстат, новости по МО и месяцам
```

## Данные и лицензии

- Данные СберИндекса (CC BY-SA 4.0) используются только для конкурса и **не входят в репозиторий**
  (Положение §9.2.2); скачиваются `make data`.
- Росстат — открытые данные со ссылкой на источник. GDELT — свободное использование со ссылкой на
  The GDELT Project. Производственный календарь — пакет `holidays` (MIT) + постановления Правительства.
- Фундаментальные модели — только Apache-2.0 (Chronos-Bolt, Chronos-2, TimesFM 2.5, TiRex-2,
  Toto-2.0). Модели с некоммерческими лицензиями (Moirai, TimesFM 3.0) не используются.
- Код решения — MIT.

При разработке кода использовались инструменты ИИ-ассистирования (Claude Code); решения и
результаты проверены автором.
