#!/usr/bin/env bash
# Download the open data of the contest and Rosstat tables into data/raw/.
# sberbank.com and rosstat.gov.ru use the Russian Trusted Root CA, which many systems do not
# trust, so curl runs with -k. Check the files by size/hash if this matters to you.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/rosstat data/raw/dict
if [ ! -f data/raw/hackathonlicence/consumption.parquet ]; then
  curl -skL -o data/raw/hackathonlicence.zip \
    https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip
  # only the parquet files: the PDF inside has a cp866 file name that some unzip builds cannot write
  unzip -o -q data/raw/hackathonlicence.zip 'hackathonlicence/*.parquet' -d data/raw/
fi
if [ ! -f data/raw/dict/t_dict_municipal_districts.xlsx ]; then
  curl -skL -o data/raw/t_dict_municipal.rar https://s.sber.ru/GthXk7
  .venv/bin/python -c "import unrar_cffi" 2>/dev/null || uv pip install --python .venv/bin/python unrar-cffi
  .venv/bin/python - <<'PY'
from unrar.cffi import rarfile
rf = rarfile.RarFile("data/raw/t_dict_municipal.rar")
for n in rf.namelist():
    if n.endswith((".xlsx", ".gpkg", ".pdf")):
        open("data/raw/dict/" + n.split("/")[-1], "wb").write(rf.read(n))
PY
fi
for f in Oborot_2025.xls Obschepit_2000-2025.xls; do
  [ -f "data/raw/rosstat/$f" ] || curl -sk -o "data/raw/rosstat/$f" "https://rosstat.gov.ru/storage/mediabank/$f"
done
# SberIndex portal: national monthly consumer spending by type (open data, sberindex.ru)
mkdir -p data/raw/portal
[ -f data/raw/portal/consumer_spending.parquet ] || curl -skL -o data/raw/portal/consumer_spending.parquet \
  "https://sberindex.ru/api/dataset/v1/download/consumer-spending/parquet"
ls -la data/raw data/raw/dict data/raw/rosstat data/raw/portal
