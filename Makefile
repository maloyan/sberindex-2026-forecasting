# СберИндекс 2026, трек «Прогнозирование».
#   Свежий клон:  make setup data quick     (проверка всех главных чисел за 1-2 мин, см. README)
#   Полный прогон: make reproduce           (4-5 ч, 33 ГБ трафика GDELT)
PY := .venv/bin/python
FM_DEVICE ?= auto

.PHONY: setup data news exog baselines prophet statsf decomp decomp-fm fm fm-local finetune cpd \
        break-aware gbm ensemble evaluate tables analysis forecast figures report events \
        decomp-v3 gbm-v3 stack ensemble-nat3 prophet-grid check-2025 \
        smoke verify quick rebuild reproduce reproduce-v2 verify-cache clean-cache

setup:            ## окружение (uv, Python 3.13), точные версии из requirements.lock.txt
	uv venv .venv --python 3.13
	uv pip install --python $(PY) -r requirements.lock.txt

data:             ## данные СберИндекса, справочник МО, Росстат, портал СберИндекса (≈ 50 МБ)
	bash scripts/get_data.sh

# ---------------------------------------------------------------- проверка (свежий клон)
smoke:            ## тесты без данных: утечки (прогноз в origin o не зависит от месяцев > o), форма (секунды)
	$(PY) -m pytest tests -q

verify:           ## пересчёт главных чисел из results/verify/cubes.npz (8 МБ в git) и сверка с таблицами
	$(PY) src/verify.py

quick:            ## есть results/cache -> rebuild; иначе smoke + verify (свежий клон, нужен make data)
	@if [ -f results/cache/ensemble.npz ]; then $(MAKE) rebuild; else $(MAKE) smoke verify; fi

rebuild:          ## по готовому кэшу: ансамбль, метрики, таблицы, прогноз, рисунки, отчёт (минуты)
	$(PY) src/ensemble.py
	$(PY) src/evaluate.py
	$(PY) src/tables.py
	$(PY) src/analysis.py
	$(PY) src/forecast_final.py
	$(PY) src/check_2025.py
	$(PY) src/figures.py
	bash scripts/build_report.sh

verify-cache:     ## пересобрать results/verify/cubes.npz из results/cache
	$(PY) src/verify.py --build

# ---------------------------------------------------------------- полный прогон
news:             ## GDELT GKG 2022-12..2024-12 (≈ 33 ГБ трафика, ~1-2 ч) -> data/external/news_*.parquet
	$(PY) src/news/gdelt_fetch.py --start 2022-12-01 --end 2024-12-31 --out data/raw/gdelt --workers 8
	$(PY) src/news/aggregate.py

exog:
	$(PY) src/exog/rosstat.py

baselines: exog
	$(PY) src/models/baselines.py
	$(PY) src/models/lgbm_v1.py

prophet:          ## ~50 мин на 12 ядрах
	$(PY) src/models/prophet_model.py --n-jobs 8

statsf:
	$(PY) src/models/statsf.py

decomp:
	$(PY) src/models/decomp.py

decomp-fm:        ## повторно: собирает прогнозы FM на локальной компоненте в рубли (после fm-local, finetune)
	$(PY) src/models/decomp.py

fm:               ## zero-shot на уровнях
	$(PY) src/fm/run_fm.py --space level --device $(FM_DEVICE)

fm-local:         ## zero-shot на локальной компоненте (после decomp)
	$(PY) src/fm/run_fm.py --space local --device $(FM_DEVICE)

finetune:         ## дообучение Chronos-2 (каузально, по отсечкам)
	$(PY) src/fm/finetune_chronos2.py

cpd:              ## бенчмарк детекторов сдвигов (синтетика, реальные данные, новости)
	$(PY) src/cpd/benchmark.py

events:           ## реестр ЧС (data/external/event_registry.csv): event study паводка + детекторы
	$(PY) src/event_study.py

break-aware:      ## сезонная наивная с поправкой после сдвига (нужен cpd)
	$(PY) src/models/break_aware.py

gbm:              ## абляция групп признаков + финальные GBM
	$(PY) src/models/gbm.py --name gbm_base --groups base calendar
	$(PY) src/models/gbm.py --name gbm_news --groups base calendar news
	$(PY) src/models/gbm.py --name gbm_cpd --groups base calendar cpd
	$(PY) src/models/gbm.py --name gbm_fm --groups base calendar fm
	$(PY) src/models/gbm.py --name gbm_full --groups base calendar fm news cpd
	$(PY) src/models/gbm.py --name cat_full --algo catboost --groups base calendar fm news cpd

ensemble:         ## ансамбль v1 (заголовок): пул из 35 моделей зафиксирован в configs/config.yaml
	$(PY) src/ensemble.py

evaluate:
	$(PY) src/evaluate.py

tables:           ## таблица выбора модели, интервалы (покрытие, WIS), таблица решений по детекторам
	$(PY) src/tables.py

analysis:
	$(PY) src/analysis.py
	$(PY) src/analysis.py --fm-contribution

forecast:
	$(PY) src/forecast_final.py

check-2025:       ## прогноз 2025 против факта портала СберИндекса
	$(PY) src/check_2025.py
	$(PY) src/check_2025.py --cubes ensemble_final ensemble_v2_final ensemble_nat3_final

figures:
	$(PY) src/figures.py

report:           ## report/report.pdf, presentation/index.html, presentation/slides.pdf
	bash scripts/build_report.sh

reproduce: data exog news baselines prophet statsf decomp fm fm-local finetune decomp-fm cpd events break-aware gbm ensemble evaluate tables analysis forecast check-2025 figures report

# ---------------------------------------------------------------- пост-hoc эксперименты (не заголовок)
decomp-v3:        ## национальная компонента v2 (Росстат по категориям) и v3 (+ портал СберИндекса)
	$(PY) src/exog/portal.py
	$(PY) src/models/decomp.py --version v2
	$(PY) src/models/decomp.py --version v3

gbm-v3:           ## GBM на национальной v3 (после decomp-v3)
	$(PY) src/models/gbm.py --name gbm3_full --groups base calendar fm news cpd --national v3 --seeds 3
	$(PY) src/models/gbm.py --name cat3_full --algo catboost --groups base calendar fm news cpd --national v3

stack:            ## ансамбль v2: все rolling origin до 2024-06, forward-выбор (≈ 15 мин)
	$(PY) src/stack.py

ensemble-nat3:    ## ансамбль v1 + декомпозиция на национальной v3 (тот же отбор на validation)
	$(PY) src/ensemble.py --variant ensemble_nat3

prophet-grid:     ## 10 конфигураций Prophet на 400 МО (≈ 2-3 ч)
	$(PY) src/models/prophet_grid.py --n-jobs 8

reproduce-v2: decomp-v3 gbm-v3 stack ensemble-nat3 prophet-grid evaluate tables check-2025

clean-cache:
	rm -rf results/cache data/interim
