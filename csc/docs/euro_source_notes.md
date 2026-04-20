# EURO (ECMWF WAM) source audit — partitioned-swell options

**Goal**: replace Open-Meteo's combined-Hs-only `ecmwf_wam025` endpoint
with a source that publishes the three ECMWF WAM partition fields
(`VHM0_SW1`, `VHM0_SW2`, `VHM0_WW`) so CSC EURO features are no longer
contaminated by wind chop.

## Summary of options

| Source | Partitions? | History? | Free? | Point-fetch? | Verdict |
|---|---|---|---|---|---|
| **Copernicus Marine (CMEMS)** | Yes — `VHM0_SW1/SW2/WW` | 1993 → now (MY) + realtime (ANFC) | Yes, free registration | Yes, via `copernicusmarine subset` | **Primary choice** |
| ECMWF Open Data | No partitions in the publicly released variable set | 4-day retention | Yes | GRIB bulk only | Reject — no partitions |
| TIGGE (ECMWF) | Ensemble only; forecast HS only, no partitions | 2006 → now | Research-only | ECMWF Web API | Reject — no partitions |
| EMODnet Physics ERDDAP | Aggregates in-situ buoys; doesn't re-stream WAM | n/a | Yes | Yes | Not useful here |
| PacIOOS / NERACOOS ERDDAP | Hosts WW3 + regional models, not ECMWF WAM | — | Yes | Yes | Not ECMWF |
| Surfline LOLA | Proprietary partitioned blend; not open | n/a | No | API requires contract | Reject |

## Chosen products

Two CMEMS products cover the full CSC training window without paywall:

### `cmems_mod_glo_wav_my_0.2deg_PT3H-i`
- "Global Ocean Waves Reanalysis" (Multi-Year)
- 0.2° grid, 3-hourly, variables include `VHM0`, `VHM0_SW1`, `VHM0_SW2`,
  `VHM0_WW`, `VTPK`, `VTM01_SW1/SW2`, `VMDR`, `VMDR_SW1/SW2`
- Coverage: 1993 → about 6–12 months behind real time
- This is the **primary source** for historical backfill from 2017
  forward.

### `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i`
- "Global Ocean Waves Analysis and Forecast" (operational)
- 0.083° grid (finer than MY), 3-hourly
- Same partition fields
- Covers the recent months where MY hasn't caught up yet; also used as
  the operational feed going forward if we want live CMEMS-sourced
  features.

`csc/cmems_backfill.py` tries MY first per year and falls back to ANFC
if MY is missing or thin. Rows are de-duped on `valid_utc` with MY
preferred over ANFC where they overlap.

## Access — what the user must do (one time)

The `copernicusmarine` Python package requires a free Copernicus Marine
account.

**If you have not already signed up:**
1. Visit <https://marine.copernicus.eu/> and click "Register" (top right).
2. Verify your email.
3. From the colesurfs conda env, run:
   ```
   copernicusmarine login
   ```
   Enter the username + password you just registered. This writes
   `~/.copernicusmarine/credentials.txt`, which `cmems_backfill.py`
   auto-detects.

Alternatively, export env vars:
```
export COPERNICUS_MARINE_SERVICE_USERNAME="you"
export COPERNICUS_MARINE_SERVICE_PASSWORD="xxx"
```

`cmems_backfill.py` aborts early with a clear message if neither is
present.

## Variable mapping (CMEMS → CSC canonical)

| CMEMS name | CSC column | Units | Open-Meteo equivalent |
|---|---|---|---|
| `VHM0_SW1` | `swell_wave_height` | m | `swell_wave_height` |
| `VTM01_SW1` | `swell_wave_period` | s | `swell_wave_period` |
| `VMDR_SW1` | `swell_wave_direction` | deg FROM | `swell_wave_direction` |
| `VHM0_SW2` | `secondary_swell_wave_height` | m | `secondary_swell_wave_height` |
| `VTM01_SW2` | `secondary_swell_wave_period` | s | `secondary_swell_wave_period` |
| `VMDR_SW2` | `secondary_swell_wave_direction` | deg FROM | `secondary_swell_wave_direction` |
| `VHM0` | `wave_height` | m | `wave_height` |
| `VTPK` | `wave_period` | s | `wave_peak_period` (closest match) |
| `VMDR` | `wave_direction` | deg FROM | `wave_direction` |

`VHM0_WW` (wind-sea) is intentionally not written — we don't train on
it today, and we can always re-pull if we start to.

## Output layout

```
.csc_data/euro_partitions/
  buoy=44013/year=2024/cmems.parquet
  buoy=44065/year=2024/cmems.parquet
  buoy=44097/year=2017/cmems.parquet
  …
```

One Parquet shard per (buoy, year). Resumable — existing shards are
skipped unless `--force` is passed.

## Expected row counts

3-hourly × 365 days = **2,920 rows/year** per buoy, give or take one
(leap years + timestep-at-midnight-of-Jan-1 inclusion). The backfill
logs the actual count per shard and prints a coverage table at the end.

If rows/year < ~2,500 the buoy likely sits at a masked grid point (land
adjacency for 44013 Boston is the most likely candidate). If this
happens we can widen the nearest-cell search or use a slightly offshore
proxy point.

## Verification status

The module is written and syntactically complete. **End-to-end
verification requires a CMEMS account and a live network call**, which
this agent's sandbox cannot perform. To verify locally:

```
cd /Users/cphwebserver/Documents/colesurfs
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate colesurfs
pip install copernicusmarine
copernicusmarine login        # one-time
python -m csc.cmems_backfill --buoy 44097 --years 2024
ls .csc_data/euro_partitions/buoy=44097/year=2024/
python -c "
import pandas as pd
df = pd.read_parquet('.csc_data/euro_partitions/buoy=44097/year=2024/cmems.parquet')
print(df.head())
print(df.columns.tolist())
print(len(df), 'rows')
"
```

## Fallback plan if CMEMS proves unsuitable

If for any reason CMEMS blocks us (rate limits, account trouble, or the
dataset gets paywalled), the next best option is to pull the
**partitioned WW3 product** served by NOAA's NOMADS
(`https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/…/gfswave.partitioned.*`).
That's WW3 not WAM, so it's a *different* model — acceptable only if we
can retire EURO-as-ECMWF-WAM entirely. Otherwise we live with Open-Meteo
bulk Hs and accept the known wind-chop contamination.
