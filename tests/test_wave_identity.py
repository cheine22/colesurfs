"""Golden regression test for the wave-parsing pipeline.

Protects the CSC2 byte-identity constraint (CLAUDE.md): every record
produced by waves._parse_response and waves_cmems.raw_rows_to_hourly_records
must stay exactly identical through any refactor. Golden outputs were
captured from the pre-consolidation code on 2026-07-03.

Regenerate goldens ONLY for an intentional behavior change:
    python tests/regen_golden.py
"""
import json
from datetime import datetime
from pathlib import Path

import pytest

from waves import _parse_response
from waves_cmems import raw_rows_to_hourly_records

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden():
    return _load("golden_records.json")


def test_openmeteo_gfs_identity(golden):
    got = _parse_response(_load("openmeteo_gfs_raw.json"))
    assert got == golden["openmeteo_gfs"]


def test_openmeteo_synthetic_identity(golden):
    got = _parse_response(_load("openmeteo_synthetic_raw.json"))
    assert got == golden["openmeteo_synthetic"]


def test_cmems_synthetic_identity(golden):
    rows = _load("cmems_synthetic_rows.json")
    for r in rows:
        r["utc"] = datetime.fromisoformat(r["utc"])
    got = raw_rows_to_hourly_records(rows)
    assert got == golden["cmems_synthetic"]
