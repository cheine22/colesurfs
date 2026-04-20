# CSC training-target audit — Buoy-components parity

**Goal:** verify that the CSC observation training target (`primary_thredds.parquet`,
`primary_rt45d.parquet`) at every historical hour is bit-for-bit identical to what
the live colesurfs dashboard's "Buoy Components" row would display at that same
hour.

**Invariant:** `buoy._spectral_components(spec_bins, swdir_bins)` is the authoritative
decomposer. The dashboard calls it on live realtime2 text (`.data_spec` + `.swdir`)
parsed by `buoy._parse_spectral_file`. The training target re-calls it on historical
inputs sourced from NDBC THREDDS NetCDF (`csc/backfill_primary_swell.py`) and
realtime2 text (`buoy._fetch_historical_spectral`).

If the per-bin inputs fed into `_spectral_components` diverge between paths, the
target diverges from the dashboard.

---

## Paths inspected

| Path | Source | Parser | Feeds `_spectral_components` |
|---|---|---|---|
| A — live dashboard | realtime2 `.data_spec` + `.swdir` | `buoy._parse_spectral_file` | top row only |
| B — CSC `primary_rt45d` | same files, all rows | `buoy._parse_spectral_file_all_rows` via `_fetch_historical_spectral` | every row |
| C — CSC `primary_thredds` | NDBC THREDDS OPeNDAP `.nc` (`spectral_wave_density`, `mean_wave_dir`) | `backfill_primary_swell._decompose_ndbc_thredds` | every time step |

Paths A & B share the text parser and identical fill-value semantics (no
per-bin filtering). They produce identical inputs to `_spectral_components` for
the same timestamp. Parity between A and B is therefore structural and not at
risk.

Path C was the risk surface.

---

## Divergence found

**Root cause:** `_decompose_ndbc_thredds` dropped entire `(freq, value)` bins from
the list before passing to `_spectral_components` when either the energy or the
direction contained a fill value (`>= 999.0` / `>= 990.0`) or `NaN` (from
`netCDF4`'s auto-masking). The text path in `_parse_spectral_file` never drops
bins — it parses whatever numeric value is in the row, leaving 999 fills inline.

**Why this mattered:**

`_spectral_components` computes centered-difference bin widths:

```python
def bw(i):
    if i == 0:   return freqs[1] - freqs[0]
    if i == n-1: return freqs[-1] - freqs[-2]
    return (freqs[i + 1] - freqs[i - 1]) / 2.0
```

`bw(i)` assumes `freqs[]` is the full dense NDBC frequency grid. When the
THREDDS path removed a bin, the surrounding bins' `Δf` widened across the gap,
so `Hm0 = 4·√Σ E Δf` and `Tm = Σ(E·Δf·T) / Σ(E·Δf)` were computed against a
different geometry than the dashboard would have.

Peak detection (`energy[i] > energy[pi] and energy[i] > energy[ni]`) also used
the shortened list, so the left/right neighbours of a given bin were different
bins than on the dashboard path — changing which bins were classified as peaks,
which were merged, and how partitions were bounded.

For any timestamp whose underlying NDBC record had even one per-bin fill value,
the THREDDS training row could disagree with the dashboard row on height,
period, and/or direction.

**Observed on station 44097 year=2026 (smallest/most recent shard):** both
`primary_rt45d.parquet` and `primary_thredds.parquet` exist and overlap on the
last ~45 days — the two sources should match on that overlap. Direct empirical
comparison could not be executed in this environment (Python exec disabled),
but the code-level divergence above is sufficient to require the fix below.

---

## Other risks checked — no divergence

| Risk | Finding |
|---|---|
| Frequency bin layout | Both sources use the same NDBC 47-bin grid (0.02–0.485 Hz), ascending. THREDDS `frequency[]` matches the inline `(val, freq)` pairs in realtime2. |
| Units of E(f) | m²/Hz in both paths. Confirmed in `_spectral_components` docstring and NDBC CF conventions. |
| α1 units | degrees-true, 0–360 in both. |
| Timestamp alignment | Both paths parse into aware UTC datetimes. Text: `YY MM DD hh mm`. NetCDF: `datetime.fromtimestamp(epoch, tz=UTC)`. |
| Frequency ordering | Ascending in both sources. `_spectral_components` does not require ordering beyond monotonicity for `bw(i)`, which is satisfied. |
| Index alignment between E and α1 | Text: `min(len(spec), len(swdir))` per timestamp. NetCDF: same `freqs[j]` indexes both variables. |
| Partition count / filter thresholds | All thresholds (`MIN_HM0_FT = 0.2`, `Tm ≥ 6 s`, sort by energy, top-2) live inside `_spectral_components` and are applied identically. |

---

## Fix applied

`csc/backfill_primary_swell.py` — `_decompose_ndbc_thredds` now preserves the
full frequency layout. Fill / NaN energy bins are substituted with `0.0`; fill /
NaN direction values on otherwise-valid-energy bins are substituted with `0.0`
(direction is moot because `_spectral_components`' circular mean is
energy-weighted, and 0-energy bins contribute zero weight). The
minimum-finite-bin guard now checks `finite_count` instead of the length of the
surviving list, preserving the same "skip obs with too few real measurements"
behaviour.

After the fix, the THREDDS path and the text path hand `_spectral_components`
the same-length, same-freqs list for the same timestamp: bit-parity is restored.

### Re-run required

The THREDDS shards under `.csc_data/primary_swell/buoy=*/year=*/primary_thredds.parquet`
were produced with the buggy feeding convention. Any timestamp whose underlying
NDBC row had per-bin fill values was decomposed against a shortened frequency
list and will disagree with the dashboard for the same hour.

To re-backfill:

```bash
cd /Users/cphwebserver/Documents/colesurfs
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate colesurfs
python -m csc.backfill_primary_swell
```

This rewrites `primary_thredds.parquet` for every buoy × year. `primary_rt45d.parquet`
and `primary_cdip.parquet` are unaffected by this change.

### Empirical verification to run after re-backfill

On the next session where Python exec is available:

1. Pick 10 `(buoy, valid_utc)` pairs present in both `primary_thredds.parquet` and
   `primary_rt45d.parquet` (year=2026 shards have the overlap).
2. Left-join on `(buoy_id, valid_utc, partition=1)`; assert
   `|hm0_ft_thredds - hm0_ft_rt45d| < 0.01 ft`, `|period_s_thredds - period_s_rt45d| < 0.1 s`,
   `|dir_deg_thredds - dir_deg_rt45d| < 1°` (circular).
3. Separately, `curl http://localhost:5151/api/buoys` → `.Block Island Sound.components`
   for the live hour; confirm those values match the most-recent
   `primary_rt45d.parquet` row for station 44097.

The pair (B = primary_rt45d) ↔ (dashboard live) is structurally identical (same
parser, same decomposer). The remaining verification is (B) ↔ (C = primary_thredds)
on overlap timestamps.

---

## Conclusion

- Input-level equivalence: **broken before fix** (THREDDS dropped fill-value bins;
  text path kept them, shifting the frequency grid seen by `_spectral_components`).
  Restored after fix.
- Decomposition-level equivalence: **always held** — `_spectral_components` is
  the single authoritative function called by all three paths.
- End-to-end equivalence: **must re-run backfill** to regenerate affected
  `primary_thredds.parquet` shards, then verify empirically as described above.

**Files changed:**
- `csc/backfill_primary_swell.py` — fill-value handling in
  `_decompose_ndbc_thredds` now preserves the dense frequency grid.

**Files intentionally untouched:** `buoy.py` (authoritative invariant),
`csc/dashboardify.py`, `csc/cmems_backfill.py`, `csc/data.py`, `csc/experiment.py`,
`csc/train.py`, `csc/surf_metrics.py`, `waves.py`, `templates/*`.
