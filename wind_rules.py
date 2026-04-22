"""colesurfs — Wind Categorization Rules

Source of truth for spot wind-rating thresholds is
    wind-categorization-scheme.toml

Mirrors swell_rules.py: `load_config(force=False)` returns the parsed
config, `reload()` re-reads from disk, `categorize(speed, dir, shore_normal,
gust)` returns one of the six ratings. `to_payload()` emits the same
structure the frontend consumes so `/api/config` and the tuner page share
a single schema.
"""

from __future__ import annotations

import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

# Evaluation order (hierarchical — first match wins).
RATINGS = ["Glassy", "Groomed", "Clean", "Textured", "Messy", "Blown Out"]

# Color tiers match swell_rules.COLORS' conventions (no frontend coupling
# needed; frontend already hardcodes these four tiers).
TIER_OF = {
    "Glassy":   "clean", "Groomed":  "clean", "Clean":    "clean",
    "Textured": "gold",
    "Messy":    "blue",
    "Blown Out": "grey",
}

_TOML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "wind-categorization-scheme.toml")
_CACHE: dict | None = None


def load_config(force: bool = False) -> dict:
    """Load sustained-wind thresholds. Legacy `*_gust_max` fields are
    accepted for backward compatibility (value read as sustained)."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    with open(_TOML_PATH, "rb") as f:
        raw = tomllib.load(f)

    def _first(d: dict, *keys) -> float:
        for k in keys:
            if k in d:
                return float(d[k])
        raise KeyError(f"none of {keys} in {d}")

    offshore = raw.get("offshore", {})
    sideshore = raw.get("sideshore", {})
    onshore  = raw.get("onshore", {})
    _CACHE = {
        "angles": {
            "offshore_max":  float(raw["angles"]["offshore_max"]),
            "sideshore_max": float(raw["angles"]["sideshore_max"]),
        },
        "low_sustained": {
            "clean_max": float(raw["low_sustained"]["clean_max"]),
        },
        "offshore": {
            "glassy_sust_max":       _first(offshore, "glassy_sust_max", "glassy_gust_max"),
            "groomed_sustained_min": _first(offshore, "groomed_sustained_min"),
        },
        "sideshore": {
            "textured_sust_max": _first(sideshore, "textured_sust_max", "textured_gust_max"),
            "messy_sust_max":    _first(sideshore, "messy_sust_max",   "messy_gust_max"),
        },
        "onshore": {
            "textured_sust_max": _first(onshore, "textured_sust_max", "textured_gust_max"),
            "messy_sust_max":    _first(onshore, "messy_sust_max",   "messy_gust_max"),
        },
    }
    return _CACHE


def reload() -> dict:
    return load_config(force=True)


def to_payload() -> dict:
    """The shape served by /api/config and consumed by the tuner page."""
    return {"wind_rating": load_config()}


def _direction_band(direction_deg: float | None,
                     shore_normal: float | None,
                     cfg: dict) -> str | None:
    if direction_deg is None or shore_normal is None:
        return None
    offshore_from = (shore_normal + 180.0) % 360.0
    d = abs((direction_deg - offshore_from) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    if d <= cfg["angles"]["offshore_max"]:
        return "offshore"
    if d <= cfg["angles"]["sideshore_max"]:
        return "sideshore"
    return "onshore"


def categorize(speed_mph: float | None,
                direction_deg: float | None,
                shore_normal: float | None,
                gust_mph: float | None = None) -> str | None:
    """Return one of RATINGS, or None if speed/shore_normal missing.
    `gust_mph` is accepted for API compatibility but ignored as of the
    2026-04-21 sustained-only migration — the matrix in /tuner is
    authoritative."""
    if speed_mph is None or shore_normal is None:
        return None
    cfg = load_config()
    band = _direction_band(direction_deg, shore_normal, cfg)

    # 1. Glassy — offshore + low sustained
    if band == "offshore" and speed_mph < cfg["offshore"]["glassy_sust_max"]:
        return "Glassy"
    # 2. Groomed — offshore + high sustained
    if band == "offshore" and speed_mph > cfg["offshore"]["groomed_sustained_min"]:
        return "Groomed"
    # 3a. Clean — offshore any speed
    if band == "offshore":
        return "Clean"
    # 3b. Clean — any direction, very light
    if speed_mph < cfg["low_sustained"]["clean_max"]:
        return "Clean"
    if direction_deg is None:
        return None
    # 4. Textured — moderate sideshore or light onshore
    if band == "sideshore" and speed_mph < cfg["sideshore"]["textured_sust_max"]:
        return "Textured"
    if band == "onshore"   and speed_mph < cfg["onshore"]["textured_sust_max"]:
        return "Textured"
    # 5. Messy — stronger sideshore or moderate onshore
    if band == "sideshore" and speed_mph < cfg["sideshore"]["messy_sust_max"]:
        return "Messy"
    if band == "onshore"   and speed_mph < cfg["onshore"]["messy_sust_max"]:
        return "Messy"
    # 6. Blown Out — everything else
    return "Blown Out"
