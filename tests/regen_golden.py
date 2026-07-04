"""Regenerate golden_records.json from the current parser code.

Run only when a wave-parsing behavior change is intentional; the diff of
golden_records.json is the review artifact for that change.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from waves import _parse_response
from waves_cmems import raw_rows_to_hourly_records

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


def main():
    golden = {
        "openmeteo_gfs": _parse_response(_load("openmeteo_gfs_raw.json")),
        "openmeteo_synthetic": _parse_response(_load("openmeteo_synthetic_raw.json")),
    }
    rows = _load("cmems_synthetic_rows.json")
    for r in rows:
        r["utc"] = datetime.fromisoformat(r["utc"])
    golden["cmems_synthetic"] = raw_rows_to_hourly_records(rows)

    with open(FIXTURES / "golden_records.json", "w") as f:
        json.dump(golden, f, indent=1, sort_keys=True)
    for k, v in golden.items():
        print(f"{k}: {len(v)} records")


if __name__ == "__main__":
    main()
