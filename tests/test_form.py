"""The contest form accepts at most 4096 characters, only ASCII and Cyrillic (no « » — × ² → ≥ −)."""
import re
from pathlib import Path

FORM = Path(__file__).resolve().parents[1] / "report" / "form_description.txt"


def bad_chars(text: str) -> set[str]:
    return {c for c in text if not (ord(c) < 128 or re.match(r"[Ѐ-ӿ]", c))}


def test_form_description_fits_the_form():
    text = FORM.read_text(encoding="utf-8")
    assert len(text) <= 4096, len(text)
    assert not bad_chars(text), sorted(bad_chars(text))
