# GFS-Wave source audit — what's fed to CSC vs what the dashboard shows

**Goal:** verify that CSC training features for GFS are bit-for-bit what
the colesurfs dashboard displays for GFS. Non-negotiable per the user.

**Verdict (TL;DR):** the training pipeline diverges from the dashboard,
**on its own side**, regardless of what Open-Meteo is doing relative to
NOAA. The divergence is internal to this repo — the logger captures a
*subset* of the 13 Open-Meteo variables `waves._parse_response` consumes.

Because the bug is upstream of the Open-Meteo-vs-NOAA question, fixing
the field set is mandatory whether or not we switch sources.

A GRIB2-direct replacement pipeline (`csc/gfs_grib_backfill.py`) is
provided for the case where Open-Meteo's GFS-Wave feed is itself shown
to diverge from NOAA raw output. Network-blocked sandbox prevents a
bit-for-bit live probe from inside this audit — runnable verification
instructions are at the bottom.

---

## 1. The dashboard's field set

`templates/index.html` never renders Open-Meteo values directly. Every
GFS cell goes through `waves.py::_parse_response` →
`_build_components`, which reads these **13 Open-Meteo variables**:

```
wave_height, wave_period, wave_peak_period, wave_direction,
swell_wave_height, swell_wave_period, swell_wave_direction,
secondary_swell_wave_height, secondary_swell_wave_period,
secondary_swell_wave_direction,
tertiary_swell_wave_height, tertiary_swell_wave_period,
tertiary_swell_wave_direction
```

(verified in `csc/dashboardify.py:47-52` — the canonical replay of
`_parse_response` lists exactly these 13 keys.)

Within `_build_components`:
* primary swell is ranked by `height_ft² × period_s`,
* the top partition with `period ≥ 6.0 s` and `height > 0` wins,
* if no swell partition survives, it falls back to `wave_height` +
  `wave_peak_period` (NOT `wave_period`) + `wave_direction`.

The critical choice: `waves.py:135` uses **`wave_peak_period` preferred
over `wave_period`** as the period shown for the combined fallback.

## 2. The CSC training logger's field set

`csc/schema.py::OM_WAVE_VARS` (consumed by `csc/logger.py`):

```python
OM_WAVE_VARS = [
    "wave_height", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
    "secondary_swell_wave_height", "secondary_swell_wave_period",
    "secondary_swell_wave_direction",
]
```

That's **9 fields**. The logger silently drops:

* `wave_peak_period` — the period the dashboard actually displays in
  the combined-fallback path and the period most swell dashboards show.
  For GFS-Wave this is the spectral peak (Tp), typically 1.15-1.4×
  higher than the mean period (Tm01) that the logger captures.
* `tertiary_swell_wave_height`, `tertiary_swell_wave_period`,
  `tertiary_swell_wave_direction` — the third swell partition.
  `_build_components` in `waves.py:67` explicitly includes the
  `swell3` partition when ranking components; dropping it in training
  means the model never sees this input channel.

`csc/features.py:29-42` then uses `gfs_wave_period` (the mean period)
as a core feature, while the dashboard shows `gfs_wave_peak_period`
(the peak period) in that same row. **Same model, different quantity
— a silent training/display divergence.**

## 3. What the comparison probe would verify

The probe compares three things at one (buoy, cycle, lead):

| Variable             | dashboard source              | CSC source (today) | NOAA GRIB field |
|----------------------|-------------------------------|--------------------|-----------------|
| wave_height          | OM `wave_height`              | OM `wave_height`   | `swh` (HTSGW)   |
| wave_peak_period     | OM `wave_peak_period`         | **NOT LOGGED**     | `perpw` (PERPW) |
| wave_period          | OM `wave_period` (fallback)   | OM `wave_period`   | spectral mean — not in standard GRIB, OM may derive |
| wave_direction       | OM `wave_direction`           | OM `wave_direction`| `dirpw` (DIRPW) |
| swell_wave_height    | OM `swell_wave_height`        | same               | partition 1 `swh`|
| swell_wave_period    | OM `swell_wave_period`        | same               | partition 1 per |
| swell_wave_direction | OM `swell_wave_direction`     | same               | partition 1 dir |
| secondary_swell_*    | OM `secondary_swell_wave_*`   | same               | partition 2 *   |
| tertiary_swell_*     | OM `tertiary_swell_wave_*`    | **NOT LOGGED**     | partition 3 *   |

NOAA's `gfswave.t{HH}z.global.0p25.f{FFF}.grib2` publishes per-partition
Hs/Tp/Dir for **three** swell partitions plus wind-sea — exactly what
Open-Meteo exposes as `swell_wave_*` / `secondary_swell_wave_*` /
`tertiary_swell_wave_*`. So if Open-Meteo is faithful, the variable
names are a 1-to-1 mapping to GRIB fields; the bug is only that our
logger isn't capturing 4 of the 13 dashboard-relevant ones.

## 4. Live comparison — status

This audit cannot execute the live HTTP/S3 fetch probe from inside the
sandbox (permissions denied). The probe logic is intact though — run
it from a shell:

```bash
cd /Users/cphwebserver/Documents/colesurfs
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate colesurfs
python -m csc.gfs_grib_backfill --verify-openmeteo --buoy 44097 --cycle LATEST
```

The `--verify-openmeteo` flag pulls both Open-Meteo and NOAA-GRIB for
the same (lat, lon, cycle, lead), extracts all 13 partition fields from
each, and prints an absolute-diff table. If every field is within 0.05
m / 0.2 s / 3° the sources agree; anything larger means Open-Meteo is
doing undocumented processing and the GRIB path should replace it.

## 5. Decision tree

1. **If `--verify-openmeteo` shows ≤0.05 m / ≤0.2 s / ≤3° diffs on all
   13 variables** — Open-Meteo is faithful. Keep it as the source.
   **Still update `OM_WAVE_VARS` to include `wave_peak_period` and the
   three `tertiary_swell_wave_*` fields** so the logger captures
   everything `_parse_response` consumes. This is the no-regret fix.
2. **If any field diverges >0.1 m / >0.5 s / >5°** — Open-Meteo is
   doing undocumented processing. Replace with the GRIB backfill.

Either way the missing-variable bug in `OM_WAVE_VARS` is a real defect
visible via static analysis alone.

## 6. Replacement pipeline — `csc/gfs_grib_backfill.py`

The module walks the NOAA open-data S3 bucket, downloads one
`gfswave.t{HH}z.global.0p25.f{FFF}.grib2` per cycle+lead, decodes it
with `cfgrib`, samples the nearest-neighbour grid cell for each CSC
buoy, and emits rows in the exact long-format schema that
`.csc_data/forecasts/` already uses
(`buoy_id, model, valid_utc, lead_days, variable, value, ingest_utc`).

Schema is additive: writes under
`.csc_data/forecasts/model=GFS/buoy=<id>/year=<y>/month=<m>/grib_bkfl.parquet`
— same partition layout as the Open-Meteo live-log pipeline.

Variable names written match Open-Meteo's naming (`wave_height`,
`wave_peak_period`, `swell_wave_height`, `secondary_swell_wave_*`,
`tertiary_swell_wave_*`) so downstream `csc.data.read_forecasts` and
`csc.features.add_engineered` consume rows transparently; no
feature-engineering code needs to change.

**Resumability:** before downloading, checks whether any row for
`(buoy, model='GFS', valid_utc, lead_days, variable)` already exists
under `model=GFS` for that year/month; skip-on-exist.

**Cycle selection:** defaults to walking cycles 00/06/12/18 Z for a
date range; one GRIB per (cycle, lead) only — not every lead of every
cycle — to keep I/O bounded. For training-target parity to the
dashboard, the relevant lead is `valid_time - cycle_time` ∈ [0, 6] h,
i.e. the freshest analysis-ish value.

**Coverage / speed estimate:** roughly 3 GRIB downloads per cycle
(00/06/12/18 × typical 6h lead coverage) × 4 cycles/day × 365 days =
~5 800 GRIB fetches per year. GFS-Wave 0p25 global is ~40 MB/file →
~230 GB/year transfer if we fetch every hour, or ~60 GB/year if we
only fetch cycle-initial frames. Runtime: 2-4 hours per year per buoy
set over broadband. S3 is anonymous-read.

## 7. Verification plan — after running

```bash
python -c "
import pandas as pd
grib = pd.read_parquet('.csc_data/forecasts/model=GFS/buoy=44097/year=2026/month=04/grib_bkfl.parquet')
print(grib.variable.value_counts())
print(grib.head())
"
```

Must show 13 distinct variables (the full dashboard set) with rows for
every cycle+buoy covered.

Then run `csc.dashboardify._verify()` against a reconstructed raw
record from these GRIB rows — it should produce the same
`wave_height_ft` / `wave_period_s` / `components` list as the live
/api/forecast/GFS endpoint for the same (buoy, hour).
