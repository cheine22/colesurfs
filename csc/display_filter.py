"""Public-facing scope filter for the /csc dashboard.

Single source of truth for which buoys and trained variants are surfaced to
the public view. West Coast training and evaluation still run silently in
the background (see csc.experiment, csc.continuous_eval) — this module
only controls what the HTTP endpoints and the dashboard template expose.

East Coast buoys are the public scope. West Coast buoys are kept in the
training pipeline for cross-coast generalization signal but hidden from
every public endpoint and template slot.
"""

from __future__ import annotations


# v3 public scope: only the 3 East Coast buoys with ≥2 years of historical
# spectral data. Barnegat (44091, deployed 2025-06) and Jeffrey's Ledge
# (44098, deployed 2025-08) are tracked in `csc.schema.FUTURE_EAST_BUOYS`
# and will be promoted here after their archives cross 24 months.
PUBLIC_BUOYS: set[str] = {"44013", "44065", "44097"}

# Hidden trained variants — anything obviously west-only. Matching is
# substring-based for the date-tagged globalboost variants so new west
# builds don't leak in as the promote process rotates tags.
HIDE_VARIANTS: set[str] = {
    "lgbm_west",
    "csc_globalboost_west_2026-04-20",
}

# Substrings that, when present in a variant name, flag it as west-only.
# Extend this if new west-specialist artifact names are introduced.
_HIDE_SUBSTRINGS: tuple[str, ...] = ("_west", "west_")


def is_public_buoy(buoy_id: str | int) -> bool:
    """True iff buoy_id should be shown on the public dashboard."""
    return str(buoy_id) in PUBLIC_BUOYS


def is_public_variant(name: str) -> bool:
    """True iff a trained variant/artifact name should be surfaced publicly."""
    if not name:
        return True
    if name in HIDE_VARIANTS:
        return False
    low = name.lower()
    for sub in _HIDE_SUBSTRINGS:
        if sub in low:
            return False
    return True


def filter_public_buoys(buoy_ids) -> list[str]:
    """Return the subset of buoy_ids that are public, preserving order."""
    return [str(b) for b in buoy_ids if is_public_buoy(b)]
