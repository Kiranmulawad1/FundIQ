"""German-aware text/date/amount normalizers for scraped grant pages.

Why a dedicated module:
  Each portal renders dates and money in a different format ("15. März
  2026", "31.12.2025", "1,5 Mio. €", "EUR 500.000"). Centralising the
  parsing here means scrapers stay focused on *finding* the values; one
  place owns *interpreting* them — and one place gets unit tests.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------
# Whitespace + boilerplate
# ---------------------------------------------------------------------------
_WHITESPACE = re.compile(r"\s+")
_SOFT_HYPHEN = "­"  # invisible hyphen used in German typography


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip; remove soft hyphens.

    German PDFs and many CMS pages drop U+00AD soft hyphens at line
    breaks. Removing them keeps embeddings and full-text indexes clean.
    """
    if not text:
        return ""
    return _WHITESPACE.sub(" ", text.replace(_SOFT_HYPHEN, "")).strip()


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------
_GERMAN_MONTH_MULTIPLIERS: dict[str, int] = {}  # filled in `parse_amount_eur`

_AMOUNT_RE = re.compile(
    r"""
    (?:€|EUR|euro)?\s*       # optional currency prefix
    (?P<num>[\d.,]+)         # digit + grouping separators
    \s*
    (?P<scale>
        million(?:en)? | mio\.? | mrd\.? | milliard(?:en)? | thousand | tsd\.? | k\b
    )?
    \s*
    (?:€|EUR|euro)?          # optional currency suffix
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SCALE_MULTIPLIERS = {
    "million": 1_000_000,
    "millionen": 1_000_000,
    "mio": 1_000_000,
    "mio.": 1_000_000,
    "mrd": 1_000_000_000,
    "mrd.": 1_000_000_000,
    "milliard": 1_000_000_000,
    "milliarden": 1_000_000_000,
    "thousand": 1_000,
    "tsd": 1_000,
    "tsd.": 1_000,
    "k": 1_000,
}


def parse_amount_eur(text: str) -> Decimal | None:
    """Parse '€ 1.500.000', '1,5 Mio. €', 'EUR 500k', '500 000 €' → Decimal.

    Rules:
      - German thousands separator is '.', decimal is ','. We invert.
      - Pure '500 000' (space-grouped) also handled.
      - Unknown / unparseable input returns None — never raises.
    """
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    m = _AMOUNT_RE.search(cleaned.lower())
    if m is None:
        return None

    num_raw = m.group("num").replace(" ", "")

    # If the string contains both '.' and ',', '.' is thousands, ',' is decimal.
    # If only '.', treat as German thousands when it groups in 3s; otherwise decimal.
    # If only ',', treat as decimal.
    if "." in num_raw and "," in num_raw:
        num = num_raw.replace(".", "").replace(",", ".")
    elif "," in num_raw:
        num = num_raw.replace(",", ".")
    elif "." in num_raw:
        # ambiguous: '1.500' could be 1500 (DE) or 1.5 (EN). Disambiguate by
        # grouping: if every '.' is followed by exactly 3 digits, it's thousands.
        parts = num_raw.split(".")
        if all(len(p) == 3 for p in parts[1:]):
            num = num_raw.replace(".", "")
        else:
            num = num_raw
    else:
        num = num_raw

    try:
        value = Decimal(num)
    except InvalidOperation:
        return None

    scale = m.group("scale")
    if scale:
        mult = _SCALE_MULTIPLIERS.get(scale.lower().rstrip("."))
        if mult:
            value *= mult

    if value < 0:
        return None
    return value


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
_GERMAN_MONTHS: dict[str, int] = {
    "januar": 1, "jan": 1,
    "februar": 2, "feb": 2,
    "märz": 3, "maerz": 3, "mrz": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mai": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10, "oct": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "dez": 12, "dec": 12,
}

# Long form: "15. März 2026" / "15 Mai 2025"
_DATE_LONG_RE = re.compile(
    r"(\d{1,2})\.?\s+([A-Za-zäöüÄÖÜ]+)\s+(\d{4})",
)

# Short form: "15.03.2026" / "15/03/2026" / "2026-03-15"
_DATE_NUMERIC_RE = re.compile(
    r"(?:(\d{4})-(\d{1,2})-(\d{1,2}))|(?:(\d{1,2})[./](\d{1,2})[./](\d{4}))",
)


def parse_german_date(text: str) -> datetime | None:
    """Parse common German date formats → timezone-aware UTC datetime.

    Returns midnight UTC on the parsed day. Time-of-day rarely matters
    for grant deadlines, and naïve datetimes have caused every off-by-
    one bug in this codebase's lineage.
    """
    if not text:
        return None
    cleaned = normalize_whitespace(text)

    # Try numeric form first — cheap and unambiguous when it matches.
    m = _DATE_NUMERIC_RE.search(cleaned)
    if m:
        if m.group(1):  # ISO 2026-03-15
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:  # DE 15.03.2026
            d, mo, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
        try:
            return datetime(y, mo, d, tzinfo=UTC)
        except ValueError:
            return None

    # Long form: "15. März 2026"
    m = _DATE_LONG_RE.search(cleaned)
    if m:
        d_raw, month_name, y_raw = m.groups()
        month_key = month_name.lower().rstrip(".")
        mo = _GERMAN_MONTHS.get(month_key)
        if mo is None:
            return None
        try:
            return datetime(int(y_raw), mo, int(d_raw), tzinfo=UTC)
        except ValueError:
            return None

    return None
