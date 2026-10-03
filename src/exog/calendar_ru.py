"""Russian production calendar features per month (known in advance, so usable as future covariates).

Public holidays and transferred days off come from the `holidays` package (MIT licence).
Working Saturdays (transfers) are not in that package; they are added from the
Government decrees on the transfer of days off:
  2024: Постановление Правительства РФ от 10.08.2023 № 1306 -> 27.04, 02.11, 28.12
  2025: Постановление Правительства РФ от 04.10.2024 № 1335 -> 01.11
Check: 247 working days in 2023, 248 in 2024, 247 in 2025 (official totals).

Features: days (calendar days), workdays, weekend_days (= days - workdays),
          holidays_n (public holidays on any weekday).
"""
from __future__ import annotations

import datetime as dt

import holidays
import pandas as pd

WORKING_SATURDAYS = {dt.date(2024, 4, 27), dt.date(2024, 11, 2), dt.date(2024, 12, 28),
                     dt.date(2025, 11, 1)}


def calendar_table(start="2023-01", end="2025-12") -> pd.DataFrame:
    rows = []
    for per in pd.period_range(start, end, freq="M"):
        hol = holidays.country_holidays("RU", years=per.year)
        days = [per.start_time.date() + dt.timedelta(i) for i in range(per.days_in_month)]
        work = [(d.weekday() < 5 and d not in hol) or d in WORKING_SATURDAYS for d in days]
        rows.append({"date": str(per), "days": per.days_in_month, "workdays": sum(work),
                     "weekend_days": per.days_in_month - sum(work),
                     "holidays_n": sum(d in hol for d in days)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = calendar_table()
    t["year"] = t.date.str[:4]
    print(t.groupby("year").workdays.sum())
