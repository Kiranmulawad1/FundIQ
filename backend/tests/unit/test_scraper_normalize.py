"""Unit tests for German-aware normalizers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.scrapers.normalize import (
    normalize_whitespace,
    parse_amount_eur,
    parse_german_date,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("  Hallo\n\tWelt  ", "Hallo Welt"),
        ("Soft­hyphen", "Softhyphen"),
        ("multi    space\n\n\nlines", "multi space lines"),
        ("", ""),
    ],
)
def test_normalize_whitespace(inp: str, expected: str) -> None:
    assert normalize_whitespace(inp) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        # German thousands separator
        ("€ 1.500.000", Decimal("1500000")),
        ("EUR 500.000", Decimal("500000")),
        # German decimal separator
        ("1,5 Mio. €", Decimal("1500000")),
        ("2,5 Mrd. EUR", Decimal("2500000000")),
        # English scale words
        ("500 thousand EUR", Decimal("500000")),
        ("3 million EUR", Decimal("3000000")),
        ("500k EUR", Decimal("500000")),
        # Plain integer
        ("3000", Decimal("3000")),
        ("3.000", Decimal("3000")),  # DE thousands form
        # Both dot and comma → '.' is thousands, ',' is decimal
        ("1.234,56 €", Decimal("1234.56")),
        # Edge: empty / garbage
        ("", None),
        ("xyz", None),
        ("--", None),
    ],
)
def test_parse_amount_eur(inp: str, expected: Decimal | None) -> None:
    assert parse_amount_eur(inp) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        # German long form
        ("15. März 2026", datetime(2026, 3, 15, tzinfo=UTC)),
        ("1 Mai 2025", datetime(2025, 5, 1, tzinfo=UTC)),
        ("31. Dezember 2026", datetime(2026, 12, 31, tzinfo=UTC)),
        # Numeric DE
        ("31.12.2026", datetime(2026, 12, 31, tzinfo=UTC)),
        ("01/06/2025", datetime(2025, 6, 1, tzinfo=UTC)),
        # ISO
        ("2026-03-15", datetime(2026, 3, 15, tzinfo=UTC)),
        # Embedded in text
        ("Antragsfrist ist der 15. März 2026 um 23:59", datetime(2026, 3, 15, tzinfo=UTC)),
        # Invalid → None, never raises
        ("", None),
        ("not a date", None),
        ("32. März 2026", None),  # invalid day
        ("15. Wurst 2026", None),  # unknown month
    ],
)
def test_parse_german_date(inp: str, expected: datetime | None) -> None:
    assert parse_german_date(inp) == expected


@pytest.mark.unit
def test_parse_german_date_returns_utc() -> None:
    """Naïve datetimes have caused every deadline off-by-one — never regress."""
    d = parse_german_date("15. März 2026")
    assert d is not None
    assert d.tzinfo is UTC
