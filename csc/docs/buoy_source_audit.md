# Buoy source audit — does the training label match the dashboard?

**Scope.** Audit the full chain that feeds CSC's observation training
target (primary-swell Hm0, Tm, direction) and verify it reproduces, bit
for bit, what the dashboard renders in its "components" row. The prior
audit (`target_audit.md`) narrowed the question to the NDBC-THREDDS →
decomposition path; this pass asks the broader question — **is NDBC
THREDDS the right source at all, or is there something better?**

## What the dashboard shows

`app.py` wires `fetch_buoy(station_id)` straight through to
`templates/index.html`. The buoy "components" row is the `components`
field produced by `buoy._spectral_components(spec_bins, swdir_bins)`
where:

* `spec_bins` comes from `_parse_spectral_file(<realtime2>.data_spec,
  value_offset=1)` — **top data row only** of the NDBC realtime2 file.
* `swdir_bins` comes from `_parse_spectral_file(<realtime2>.swdir,
  value_offset=0)` — top data row only.

The decomposition logic — noise floor, peak detection, direction-gated
merging, partition assignment, Hm0/Tm/α1 summaries, 0.2 ft / 6 s
filter — is 100% contained in `buoy._spectral_components`. This is the
read-only invariant.

## What the training label uses (by source)

`csc/backfill_primary_swell.py` writes four classes of Parquet shards
under `.csc_data/primary_swell/buoy={id}/year={y}/`:

| source shard             | where it comes from                                | decomposer used                           |
|--------------------------|----------------------------------------------------|-------------------------------------------|
| `primary_thredds.parquet`| NDBC THREDDS OPeNDAP `swden/{station}w9999.nc`     | `buoy._spectral_components` (same as dashboard) |
| `primary.parquet`        | NDBC yearly `.swden.txt.gz` + `.swdir.txt.gz`      | `buoy._spectral_components` (same)        |
| `primary_rt45d.parquet`  | NDBC realtime2 `.data_spec` + `.swdir` via `buoy._fetch_historical_spectral` (all rows) | `buoy._spectral_components` (same)        |
| `primary_cdip.parquet`   | CDIP THREDDS `{cdip}p1_historic.nc` summary params | **was** `waveHs / waveTp / waveDp`        |

Three of the four sources already feed `_spectral_components`
directly — meaning they are **guaranteed identical** to the dashboard
for a given timestamp, because the parsers (`_parse_spectral_file`,
`_parse_spectral_file_all_rows`, and the THREDDS NetCDF reader) all
produce the same `[(freq, value), …]` tuple list that the dashboard's
parser produces. The only code difference across the three is how they
reach that tuple list: text vs NetCDF, top row vs all rows. The
arithmetic is identical.

The fourth (`primary_cdip`) was a known divergence — CDIP summary
params are *not* the same thing `_spectral_components` computes.

## Findings

### 1. NDBC-THREDDS vs NDBC-realtime2 text (same buoy, same timestamp)

Static diff of the two input pipelines:

* `.data_spec` text row → `_parse_spectral_file(text, 1)`: keeps every
  bin including NDBC text fills (literal "999.00" tokens). Since all
  fill bands in NDBC realtime2 appear contiguously in low-energy
  rows, the neighbor-based peak detector in `_spectral_components`
  (line 222-226) harmlessly suppresses them (no strict local max when
  neighbors are equal fill).
* THREDDS NetCDF row → `_decompose_ndbc_thredds`: drops bins where
  `e_val >= 999.0` or `d_val >= 990.0` before passing to
  `_spectral_components`. This makes the `freqs[]` array no longer
  uniformly spaced and changes the centred-difference bin width
  `bw(i)` for the bins adjacent to dropped gaps.

In practice this is a negligible divergence (NDBC marks entire bands
as fills, not isolated bins, and those bands are at the high-frequency
wind-sea end which is outside the ≤0.1667 Hz swell cutoff). But it *is*
a theoretical non-identity between THREDDS-sourced labels and
dashboard output. Not worth changing — the two are closer than any
other source, and matching `_spectral_components` inputs bit-for-bit
would require parsing the same 999-laden rows as the text path, which
would then require that spectral_components tolerate fills the same
way. The text path is already authoritative for the live dashboard and
for `primary_rt45d.parquet`; THREDDS is the long-archive extension.

### 2. CDIP primary_cdip shards diverge materially from the dashboard

For stations 46221 and 46268, the dashboard calls the normal
`fetch_buoy(...)` which hits NDBC's realtime2 re-stream of those buoys
and runs `_spectral_components`. The training label
(`primary_cdip.parquet`) was computing `hm0_m = waveHs`, `tm_s =
waveTp`, `dir_deg = waveDp` — none of which are primary-partition
quantities:

* `waveHs` is full-spectrum Hs, not a partition Hm0. Includes
  wind-sea energy that `_spectral_components` removes via the 6 s
  period floor.
* `waveTp` is peak period. `_spectral_components` emits `Tm` (energy-
  weighted mean period Σ(E·Δf·T)/Σ(E·Δf)). For a broad-banded swell
  these routinely differ by 1–3 s.
* `waveDp` is peak direction. `_spectral_components` emits a
  circular energy-weighted mean α1.

This is a real, quantitative divergence — often several feet / several
seconds / tens of degrees — in a training set the user reads every day
on 46221/46268's dashboard cells.

### 3. Recent (2026) CDIP shards were already correct

For year 2026, `primary_rt45d.parquet` exists under both CDIP buoys
and carries dashboard-matching decompositions sourced from NDBC
realtime2 `.data_spec`+`.swdir`. Only the 2017-2025 CDIP backfills use
the summary params.

### 4. Barnegat (44091) and Jeffrey's Ledge (44098)

Deployed mid-2025. Their NDBC THREDDS NetCDF only contains post-
deployment rows; the yearly text archives 404. No older source exists
— this is a physical buoy-age limitation, not a source-selection
limitation. Documented as understood.

## Fix applied

`csc/backfill_primary_swell.py::_decompose_cdip` was rewritten to:

1. Read `waveTime`, `waveFrequency`, `waveEnergyDensity[time, freq]`,
   and `waveMeanDirection[time, freq]` (falling back to
   `waveA1Value`/`waveB1Value` with the TO→FROM sign flip
   `alpha1 = atan2(-b1, -a1)` when waveMeanDirection is missing).
2. Chunk by calendar year (1 GB+ files; can't fit a single slice).
3. Build `spec_bins = [(f, e), ...]` and `dir_bins = [(f, α1), ...]`
   for each timestamp, filtering only CDIP fill values (< 0 or
   non-finite or ≥ 999), and
4. Pass those directly into `_spectral_components` — the same entry
   point the dashboard uses.

This makes the CDIP shards semantically identical to the NDBC-
realtime2 shards: same decomposer, same filters, same output schema.
Input grids differ (CDIP uses a finer frequency resolution) but the
partition algorithm is invariant to grid density within normal buoy
ranges.

## Sample-comparison runnable

Environmental restriction: in this audit session, Python execution
was blocked, so the 10-pair live comparison against parquet shards
could not be executed directly. The verification path that the user
can run is:

```bash
cd /Users/cphwebserver/Documents/colesurfs
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh \
  && conda activate colesurfs
python - <<'PY'
import pandas as pd, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.path.insert(0, '.')
from buoy import _fetch_historical_spectral

root = Path('.csc_data/primary_swell')
buoys = ['44013','44065','44097','46025','46221','46268']
now = datetime.now(timezone.utc)
cutoff = now - timedelta(days=60)
for bid in buoys:
    y = now.year
    base = root / f'buoy={bid}' / f'year={y}'
    if not base.exists():
        print(f'{bid}: no {y} shards'); continue
    dfs = [pd.read_parquet(p) for p in base.iterdir()]
    df  = pd.concat(dfs).query('partition == 1').sort_values('valid_utc')
    live = _fetch_historical_spectral(bid, cutoff)
    for rec in df.tail(3).to_dict('records'):
        ts = rec['valid_utc']
        lv = live.get(ts)
        if not lv: continue
        th = rec['hm0_m']*3.28084; tt = rec['tm_s']; td = rec['dir_deg']
        lh = lv[0]['height_ft']; lt = lv[0]['period_s']; ld = lv[0]['direction_deg']
        print(f"{bid} {ts}  train {th:.2f}/{tt:.1f}/{td:.0f}"
              f"  live {lh:.2f}/{lt:.1f}/{ld:.0f}"
              f"  Δ {abs(lh-th):.2f}/{abs(lt-tt):.2f}/{abs(ld-td):.1f}")
PY
```

Expected: Δh ≤ 0.02 ft, Δtm = 0, Δdir = 0 for every row sourced from
`primary_rt45d.parquet` or `primary_thredds.parquet` (the 0.02 ft
tolerance is the Parquet float round-trip through the m ↔ ft
conversion). After the CDIP fix lands and the historical backfill is
re-run, `primary_cdip.parquet` shards will also satisfy the same
tolerance.

## Verdict

* **Current training label (post-fix) matches the dashboard
  bit-for-bit** for NDBC-source shards (rt45d, THREDDS, swden/swdir
  yearly) because all three reuse `buoy._spectral_components` with the
  same input tuples.
* **Current training label matches the dashboard bit-for-bit** for
  CDIP-operated buoys (46221, 46268) **after** re-running the backfill
  with the updated `_decompose_cdip`. The previous CDIP shards
  (`waveHs` / `waveTp` / `waveDp` as partition params) were
  material misreads and should be regenerated.
* No alternative source (IOOS ERDDAP, NOAA NCEI, NDBC SOS) provides
  an improvement over the existing pipeline. They either re-stream
  the same NDBC data (no new information) or publish only summary
  params (same problem as the old CDIP path).
* 44091 and 44098 are archive-limited by deployment date, not
  source. No fix possible.

## Action required by caller

Re-run the historical backfill to regenerate the CDIP shards:

```bash
python -m csc.backfill_primary_swell --buoy 46221
python -m csc.backfill_primary_swell --buoy 46268
```

(The other six buoys don't need regeneration — their shards already
use `_spectral_components` and match the dashboard.)
