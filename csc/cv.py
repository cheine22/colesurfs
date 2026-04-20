"""K-fold month-based cross-validation for CSC.

Provides two splitters:

  * `kfold_month_split`  — month-based 4-fold split that guarantees every
    meteorological season (DJF / MAM / JJA / SON) appears in every fold's
    test set. Deterministic via sorted (year, month) enumeration. Preserved
    as a simpler fallback and still used by `csc.experiment` /
    `csc.surf_metrics`.

  * `stratified_kfold_month_split` — joint stratification across season,
    observed Hs quartile (per-buoy), period band
    (`csc.schema.PERIOD_BANDS`), and 8-way direction quadrant
    (`csc.schema.DIRECTION_QUADRANTS`). Each (year, month) block is labeled
    by the dominant stratum of its rows and allocated across folds using
    iterative stratification (Sechidis et al. 2011, adapted for whole-month
    blocks). Whole-month contiguity is preserved so temporal
    autocorrelation is not leaked across folds.

Algorithm choice — Option A (iterative stratification on whole months):

  1. Tag every row with (year, month), season, per-buoy Hs-quartile bin,
     period band, and direction quadrant.
  2. For each unique (year, month), compute the dominant value of each
     stratum axis across its rows. The month's joint label is the tuple
     (season, hs_q, period_band, dir_quad).
  3. Iteratively assign months to folds: at each step pick the month whose
     label is rarest globally (breaks ties by largest row count); place it
     into whichever fold is currently most deficient in that label (ties
     broken by smallest current row count). This is a row-weighted
     round-robin that converges toward balanced stratum coverage.
  4. Fold i's test set = rows whose (year, month) was assigned to fold i.

The iterative scheme keeps whole months intact (no within-month leakage)
while ensuring that every stratum with adequate support lands in every
fold's test set. If some fold's test set is missing a stratum that has
≥20 rows globally, the report surfaces it.

Deterministic: the RNG is seeded from `seed` and affects only tie-breaking,
so re-running with the same seed reproduces the partition exactly.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from csc.schema import DIRECTION_QUADRANTS, PERIOD_BANDS, SEASONS


MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}


def _season_of(month: int) -> str:
    return MONTH_TO_SEASON[int(month)]


def _unique_year_months(df: pd.DataFrame) -> list[tuple[int, int]]:
    ym = (df["valid_utc"].dt.year.astype(int).astype(str) + "-"
          + df["valid_utc"].dt.month.astype(int).astype(str).str.zfill(2))
    ym_unique = sorted(set(ym.tolist()))
    return [(int(s.split("-")[0]), int(s.split("-")[1])) for s in ym_unique]


def _assign_fold_per_month(year_months: list[tuple[int, int]],
                           n_folds: int) -> dict[tuple[int, int], int]:
    """Round-robin each season's months across n_folds bins.

    Sorted chronologically within a season so adjacent months (Dec-2022,
    Jan-2023, Feb-2023) end up spread across distinct folds rather than
    all piling into one.
    """
    by_season: dict[str, list[tuple[int, int]]] = {s: [] for s in SEASONS}
    for ym in year_months:
        by_season[_season_of(ym[1])].append(ym)
    for s in by_season:
        by_season[s].sort()
    for s, months in by_season.items():
        if len(months) < n_folds:
            print(f"[cv] WARN: season {s} has only {len(months)} months; "
                  f"some folds will have 0 {s} test months "
                  f"(need {n_folds} for full coverage)")
    fold_of: dict[tuple[int, int], int] = {}
    for s, months in by_season.items():
        for i, ym in enumerate(months):
            fold_of[ym] = i % n_folds
    return fold_of


def kfold_month_split(df: pd.DataFrame, n_folds: int = 4
                      ) -> list[tuple[pd.DataFrame, pd.DataFrame,
                                      dict[str, Any]]]:
    """Return a list of `(train_df, test_df, fold_meta)` tuples.

    `fold_meta` contains the fold index, the test months grouped by
    season, and row counts. Seeding is implicit: the split is fully
    determined by the sorted list of (year, month) pairs, so it's
    reproducible without an explicit RNG.
    """
    if df.empty:
        raise ValueError("kfold_month_split: input frame is empty")
    year_months = _unique_year_months(df)
    if len(year_months) < n_folds:
        raise ValueError(
            f"kfold_month_split: only {len(year_months)} unique months, "
            f"need at least n_folds={n_folds}")
    fold_of = _assign_fold_per_month(year_months, n_folds)

    ym_col = list(zip(df["valid_utc"].dt.year.astype(int),
                      df["valid_utc"].dt.month.astype(int)))
    fold_col = pd.Series([fold_of[ym] for ym in ym_col], index=df.index)

    folds: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for i in range(n_folds):
        test_mask = fold_col == i
        train = df[~test_mask].reset_index(drop=True)
        test = df[test_mask].reset_index(drop=True)
        test_months = sorted(ym for ym, f in fold_of.items() if f == i)
        by_season: dict[str, list[str]] = {s: [] for s in SEASONS}
        for (y, m) in test_months:
            by_season[_season_of(m)].append(f"{y:04d}-{m:02d}")
        fold_meta = {
            "fold": i,
            "n_folds": n_folds,
            "test_months": [f"{y:04d}-{m:02d}" for (y, m) in test_months],
            "test_months_by_season": by_season,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
        folds.append((train, test, fold_meta))
    return folds


def fold_month_report(df: pd.DataFrame, n_folds: int = 4) -> dict[str, Any]:
    """Summarize what each fold's test set contains. Pure reporting — does
    not mutate or re-run the split."""
    folds = kfold_month_split(df, n_folds=n_folds)
    total = int(len(df))
    report: dict[str, Any] = {
        "n_folds": n_folds,
        "total_rows": total,
        "folds": [],
    }
    for _, test, meta in folds:
        frac = (meta["test_rows"] / total) if total else 0.0
        entry = dict(meta)
        entry["test_row_fraction"] = round(frac, 4)
        report["folds"].append(entry)
    return report


# ─── Stratified k-fold ────────────────────────────────────────────────────

DEFAULT_STRATA = ("season", "hs_q", "period_band", "dir_quad")


def _period_band_of(tp_s: float) -> str:
    if tp_s is None or (isinstance(tp_s, float) and np.isnan(tp_s)):
        return "unknown"
    for name, lo, hi in PERIOD_BANDS:
        if lo <= tp_s < hi:
            return name
    return PERIOD_BANDS[-1][0]


def _dir_quad_of(deg: float) -> str:
    """Map wave-FROM direction in degrees to an 8-way compass bucket.

    Uses 45°-wide bins centered on cardinal/intercardinal points: N covers
    (337.5, 22.5], NE (22.5, 67.5], etc. Any NaN returns 'unknown'.
    """
    if deg is None or (isinstance(deg, float) and np.isnan(deg)):
        return "unknown"
    d = float(deg) % 360.0
    idx = int(((d + 22.5) % 360.0) // 45.0)
    return DIRECTION_QUADRANTS[idx]


def _hs_quartile_per_buoy(df: pd.DataFrame) -> pd.Series:
    """Return a Series of per-row quartile labels ('Q1'..'Q4') computed
    independently per buoy_id so each buoy contributes equally to every
    bucket. Falls back to 'unknown' for rows with NaN obs_hs_m."""
    labels = pd.Series(["unknown"] * len(df), index=df.index, dtype=object)
    if "buoy_id" not in df.columns or "obs_hs_m" not in df.columns:
        return labels
    for bid, sub in df.groupby("buoy_id"):
        vals = sub["obs_hs_m"].astype(float)
        valid = vals.dropna()
        if len(valid) < 4:
            labels.loc[sub.index] = "Q_all"
            continue
        try:
            q = pd.qcut(valid, 4, labels=["Q1", "Q2", "Q3", "Q4"],
                        duplicates="drop")
        except ValueError:
            labels.loc[sub.index] = "Q_all"
            continue
        labels.loc[q.index] = q.astype(str).values
    return labels


def _tag_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a lightweight frame with columns: year, month, season, hs_q,
    period_band, dir_quad — one row per input row, aligned by index."""
    out = pd.DataFrame(index=df.index)
    out["year"] = df["valid_utc"].dt.year.astype(int)
    out["month"] = df["valid_utc"].dt.month.astype(int)
    out["season"] = out["month"].map(MONTH_TO_SEASON)
    out["hs_q"] = _hs_quartile_per_buoy(df)
    tp = df.get("obs_tp_s")
    if tp is None:
        out["period_band"] = "unknown"
    else:
        out["period_band"] = tp.astype(float).map(_period_band_of)
    dp = df.get("obs_dp_deg")
    if dp is None:
        out["dir_quad"] = "unknown"
    else:
        out["dir_quad"] = dp.astype(float).map(_dir_quad_of)
    return out


def _dominant_label(series: pd.Series) -> Any:
    """Most-frequent value in a Series (first lexicographically on ties)."""
    counts = series.value_counts()
    if counts.empty:
        return "unknown"
    top = counts.iloc[0]
    winners = sorted(counts[counts == top].index.tolist())
    return winners[0]


def _iterative_stratify_months(
    month_labels: dict[tuple[int, int], tuple[Any, ...]],
    month_rows: dict[tuple[int, int], int],
    n_folds: int,
    seed: int,
) -> dict[tuple[int, int], int]:
    """Assign each (year, month) to a fold index using iterative
    stratification over the joint label tuple.

    Inspired by Sechidis, Tsoumakas & Vlahavas (2011) "On the Stratification
    of Multi-Label Data". We adapt it for whole-month blocks:

      * Desired rows-per-label-per-fold ≈ total_rows(label) / n_folds.
      * At each step select the rarest remaining label globally
        (ties → largest-remaining-row-count). Within that label, pick the
        month with the largest row count. Allocate it to the fold currently
        most deficient for that label (ties → fold with fewest rows
        overall; further ties → deterministic RNG pick).
    """
    rng = random.Random(seed)

    remaining = set(month_labels.keys())
    # rows per label still to place
    label_rows: dict[tuple[Any, ...], int] = defaultdict(int)
    label_months: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for ym, lab in month_labels.items():
        label_rows[lab] += month_rows[ym]
        label_months[lab].append(ym)

    # desired per fold
    desired: dict[tuple[Any, ...], list[float]] = {
        lab: [label_rows[lab] / n_folds] * n_folds for lab in label_rows
    }
    # actual placed
    placed_rows_per_label: dict[tuple[Any, ...], list[int]] = {
        lab: [0] * n_folds for lab in label_rows
    }
    fold_total_rows = [0] * n_folds
    assignment: dict[tuple[int, int], int] = {}

    while remaining:
        # pick rarest label (smallest remaining total row count) that still
        # has unassigned months; ties → largest remaining-row-count (so we
        # lock in chunky labels while small ones are still findable).
        candidate_labels = [lab for lab in label_rows
                            if any(ym in remaining for ym in label_months[lab])]
        if not candidate_labels:
            break
        # remaining rows per label
        rem_rows_per_label = {
            lab: sum(month_rows[ym] for ym in label_months[lab] if ym in remaining)
            for lab in candidate_labels
        }
        min_rem = min(rem_rows_per_label.values())
        tied = [lab for lab, r in rem_rows_per_label.items() if r == min_rem]
        if len(tied) > 1:
            tied.sort(key=lambda lab: (-label_rows[lab], lab))
        lab = tied[0]

        # pick the largest remaining month under that label
        ym_candidates = [ym for ym in label_months[lab] if ym in remaining]
        ym_candidates.sort(key=lambda ym: (-month_rows[ym], ym))
        ym = ym_candidates[0]

        # pick fold most deficient for this label
        deficits = [desired[lab][f] - placed_rows_per_label[lab][f]
                    for f in range(n_folds)]
        max_def = max(deficits)
        fold_choices = [f for f in range(n_folds) if deficits[f] == max_def]
        if len(fold_choices) > 1:
            # break ties by smallest total rows already assigned
            min_total = min(fold_total_rows[f] for f in fold_choices)
            fold_choices = [f for f in fold_choices
                            if fold_total_rows[f] == min_total]
        if len(fold_choices) > 1:
            rng.shuffle(fold_choices)
        fold = fold_choices[0]

        assignment[ym] = fold
        remaining.discard(ym)
        rows = month_rows[ym]
        fold_total_rows[fold] += rows
        # every label-axis this month carries gets credited; in our case
        # the month has exactly one joint-label tuple, but we update the
        # joint label's per-fold tally here.
        placed_rows_per_label[lab][fold] += rows

    return assignment


def _stratum_coverage(train_labels: list[tuple[Any, ...]],
                      test_labels: list[tuple[Any, ...]]
                      ) -> dict[str, Any]:
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    all_strata = set(train_counts) | set(test_counts)
    present_train = sum(1 for s in all_strata if train_counts[s] > 0)
    present_test = sum(1 for s in all_strata if test_counts[s] > 0)
    missing_in_test = sorted(s for s in all_strata
                             if train_counts[s] > 0 and test_counts[s] == 0)
    return {
        "total_strata": len(all_strata),
        "train_strata_present": present_train,
        "test_strata_present": present_test,
        "test_coverage_pct": round(
            100.0 * present_test / len(all_strata), 2) if all_strata else 0.0,
        "missing_in_test": [list(s) for s in missing_in_test],
        "train_counts": {"|".join(map(str, k)): v
                         for k, v in train_counts.items()},
        "test_counts": {"|".join(map(str, k)): v
                        for k, v in test_counts.items()},
    }


def stratified_kfold_month_split(
    df: pd.DataFrame,
    n_folds: int = 4,
    strata: tuple[str, ...] = DEFAULT_STRATA,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Jointly stratified month-based k-fold split.

    Same return shape as `kfold_month_split`: a list of
    `(train_df, test_df, fold_meta)` tuples. Each month is assigned to
    exactly one fold (the test fold) using iterative stratification across
    the joint `strata` label, so every fold's test set sees a
    representative slice of every `(season × Hs-quartile × period-band ×
    direction-quadrant)` cell that has adequate support. Whole-month
    contiguity is preserved.

    Args:
      df: training frame with `valid_utc`, `buoy_id`, and observed
        `obs_hs_m` / `obs_tp_s` / `obs_dp_deg` columns.
      n_folds: number of CV folds.
      strata: which axes to stratify over. Must be a subset of
        `DEFAULT_STRATA = ('season', 'hs_q', 'period_band', 'dir_quad')`.
      seed: RNG seed for deterministic tie-breaking.

    Returns:
      list of `(train_df, test_df, fold_meta)` where `fold_meta`
      additionally contains per-fold stratum coverage stats.
    """
    if df.empty:
        raise ValueError("stratified_kfold_month_split: input frame is empty")
    unknown_axes = set(strata) - set(DEFAULT_STRATA)
    if unknown_axes:
        raise ValueError(
            f"stratified_kfold_month_split: unknown strata {unknown_axes}; "
            f"valid axes are {DEFAULT_STRATA}")

    year_months = _unique_year_months(df)
    if len(year_months) < n_folds:
        raise ValueError(
            f"stratified_kfold_month_split: only {len(year_months)} unique "
            f"months, need at least n_folds={n_folds}")

    tags = _tag_strata(df)
    for axis in strata:
        if axis not in tags.columns:
            raise ValueError(f"missing stratum column: {axis}")

    # per-row joint label (used for reporting)
    row_labels = list(zip(*[tags[a].astype(str).tolist() for a in strata]))

    # per-month dominant label (used for assignment). Group by
    # (year, month) via separate columns — pandas handles tuple-valued
    # columns inconsistently across versions.
    ym_series = list(zip(tags["year"].tolist(), tags["month"].tolist()))
    month_labels: dict[tuple[int, int], tuple[Any, ...]] = {}
    month_rows: dict[tuple[int, int], int] = {}
    for (y, m), sub in tags.groupby(["year", "month"], sort=False):
        ym = (int(y), int(m))
        lab = tuple(_dominant_label(sub[a]) for a in strata)
        month_labels[ym] = lab
        month_rows[ym] = int(len(sub))

    fold_of = _iterative_stratify_months(
        month_labels, month_rows, n_folds=n_folds, seed=seed)

    # every month must be assigned
    missing = [ym for ym in year_months if ym not in fold_of]
    if missing:
        raise RuntimeError(
            f"stratified split failed to assign months: {missing}")

    fold_col = pd.Series([fold_of[ym] for ym in ym_series], index=df.index)

    folds: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for i in range(n_folds):
        test_mask = fold_col == i
        train = df[~test_mask].reset_index(drop=True)
        test = df[test_mask].reset_index(drop=True)

        test_months = sorted(ym for ym, f in fold_of.items() if f == i)
        by_season: dict[str, list[str]] = {s: [] for s in SEASONS}
        for (y, m) in test_months:
            by_season[_season_of(m)].append(f"{y:04d}-{m:02d}")

        train_labs = [row_labels[idx] for idx, m in enumerate(test_mask) if not m]
        test_labs = [row_labels[idx] for idx, m in enumerate(test_mask) if m]
        coverage = _stratum_coverage(train_labs, test_labs)

        fold_meta: dict[str, Any] = {
            "fold": i,
            "n_folds": n_folds,
            "strata": list(strata),
            "seed": seed,
            "test_months": [f"{y:04d}-{m:02d}" for (y, m) in test_months],
            "test_months_by_season": by_season,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "stratum_coverage": coverage,
        }
        folds.append((train, test, fold_meta))
    return folds


def fold_stratum_report(
    splits: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    min_global_rows: int = 20,
) -> dict[str, Any]:
    """Summarize stratum coverage across all folds.

    Args:
      splits: output of `stratified_kfold_month_split`.
      min_global_rows: a stratum absent from a fold's test set is flagged
        as `undersampled` only if its global row count exceeds this
        threshold (default 20 per the spec).

    Returns:
      dict with overall stats, per-fold coverage, and two lists —
      `undersampled` (strata with sparse test coverage in some fold) and
      `over_represented` (strata whose test share exceeds twice the
      uniform expectation in some fold).
    """
    if not splits:
        return {"n_folds": 0, "folds": []}

    n_folds = len(splits)
    per_fold_test_counts: list[Counter] = []
    per_fold_coverage: list[dict[str, Any]] = []
    for _, _, meta in splits:
        cov = meta.get("stratum_coverage", {})
        tc = Counter(cov.get("test_counts", {}))
        per_fold_test_counts.append(tc)
        per_fold_coverage.append({
            "fold": meta["fold"],
            "total_strata": cov.get("total_strata", 0),
            "test_strata_present": cov.get("test_strata_present", 0),
            "test_coverage_pct": cov.get("test_coverage_pct", 0.0),
            "missing_in_test": cov.get("missing_in_test", []),
            "test_rows": meta.get("test_rows", 0),
            "train_rows": meta.get("train_rows", 0),
        })

    # True global tally per stratum = fold-0's train_counts + fold-0's
    # test_counts (fold-0's train already contains every other fold's test
    # rows by construction, so summing train[0] + test[0] gives the full
    # universe exactly once).
    first_meta = splits[0][2].get("stratum_coverage", {})
    global_true: Counter = Counter(first_meta.get("train_counts", {}))
    for k, v in per_fold_test_counts[0].items():
        global_true[k] += v
    global_true = Counter({k: v for k, v in global_true.items() if v > 0})

    undersampled: list[dict[str, Any]] = []
    over_represented: list[dict[str, Any]] = []
    uniform_share = 1.0 / n_folds

    for stratum, total in global_true.items():
        per_fold_counts = [tc.get(stratum, 0) for tc in per_fold_test_counts]
        fold_sum = sum(per_fold_counts)
        if fold_sum == 0:
            continue
        shares = [c / fold_sum for c in per_fold_counts]
        # undersampled: absent (count=0) in some fold while globally above
        # the threshold, OR the fold's share is less than 25% of uniform.
        for f, (c, share) in enumerate(zip(per_fold_counts, shares)):
            if total >= min_global_rows and c == 0:
                undersampled.append({
                    "stratum": stratum,
                    "fold": f,
                    "global_rows": total,
                    "test_rows_in_fold": c,
                    "reason": "absent_in_test",
                })
            elif share < 0.25 * uniform_share and total >= min_global_rows:
                undersampled.append({
                    "stratum": stratum,
                    "fold": f,
                    "global_rows": total,
                    "test_rows_in_fold": c,
                    "share": round(share, 4),
                    "reason": "share_below_quarter_uniform",
                })
            elif share > 2.0 * uniform_share and total >= min_global_rows:
                over_represented.append({
                    "stratum": stratum,
                    "fold": f,
                    "global_rows": total,
                    "test_rows_in_fold": c,
                    "share": round(share, 4),
                })

    return {
        "n_folds": n_folds,
        "total_strata": len(global_true),
        "min_global_rows_threshold": min_global_rows,
        "folds": per_fold_coverage,
        "undersampled": undersampled,
        "over_represented": over_represented,
    }
