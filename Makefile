# СберИндекс 2026, трек «Прогнозирование». Полный прогон: make reproduce
PY := .venv/bin/python
FM_DEVICE ?= auto

.PHONY: setup data exog baselines prophet statsf fm fm-local finetune decomp decomp-fm break-aware gbm ensemble evaluate analysis cpd news forecast figures report reproduce quick clean-cache

setup:            ## окружение (uv, Python 3.13)
	uv venv .venv --python 3.13
	uv pip install --python $(PY) -r requirements.lock.txt

data:             ## данные СберИндекса, справочник МО, Росстат (≈50 МБ)
	bash scripts/get_data.sh

news:             ## GDELT GKG 2022-12..2024-12 (≈33 ГБ трафика, ~1-2 ч) -> data/external/news_*.parquet
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

break-aware:      ## сезонная наивная с поправкой после сдвига (нужен cpd)
	$(PY) src/models/break_aware.py

gbm:              ## абляция групп признаков + финальные GBM
	$(PY) src/models/gbm.py --name gbm_base --groups base calendar
	$(PY) src/models/gbm.py --name gbm_news --groups base calendar news
	$(PY) src/models/gbm.py --name gbm_cpd --groups base calendar cpd
	$(PY) src/models/gbm.py --name gbm_fm --groups base calendar fm
	$(PY) src/models/gbm.py --name gbm_full --groups base calendar fm news cpd
	$(PY) src/models/gbm.py --name cat_full --algo catboost --groups base calendar fm news cpd

ensemble:
	$(PY) src/ensemble.py

evaluate:
	$(PY) src/evaluate.py

analysis:
	$(PY) src/analysis.py
	$(PY) src/analysis.py --fm-contribution

forecast:
	$(PY) src/forecast_final.py

figures:
	$(PY) src/figures.py

report:           ## report/report.pdf, presentation/index.html, presentation/slides.pdf
	bash scripts/build_report.sh

quick:            ## пересчёт метрик, ансамбля, прогноза и отчёта по готовому кэшу (минуты)
	$(PY) src/ensemble.py
	$(PY) src/evaluate.py
	$(PY) src/analysis.py
	$(PY) src/forecast_final.py
	$(PY) src/figures.py
	bash scripts/build_report.sh

reproduce: data exog news baselines prophet statsf decomp fm fm-local finetune decomp-fm cpd break-aware gbm ensemble evaluate analysis forecast figures report

clean-cache:
	rm -rf results/cache data/interim
