"""
colesurfs — shared wave-record processing used by waves.py (Open-Meteo GFS)
and waves_cmems.py (CMEMS EURO).

Behavior is locked by development-assets/tests/test_wave_identity.py — any change here must be an intentional,
golden-diff-reviewed change, because CSC2 training data must stay
byte-identical to dashboard rendering (see CLAUDE.md).
"""
import math

from config import m_to_ft

# Filter pure wind chop. 5.0 s targets real swell at Tp ~ 6 s;
# anything shorter is effectively sea, not swell.
MIN_SWELL_PERIOD_S = 5.0


def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_swell_components(raw_parts, period_scale=1.0, limit=2):
    """Filter + shape raw wave partitions into the display component list.

    raw_parts: iterable of {"h_m", "p", "d", "type"} dicts, in partition order.
    period_scale: multiplier applied to the raw period before display/energy
    (CMEMS passes 1.20 to convert Tm01 → Tp; Open-Meteo periods are already Tp).

    Returns the top `limit` components by energy (h_ft² × Tp); limit=None
    returns the whole ranking.
    """
    comps = []
    for c in raw_parts:
        h_m = safe_float(c["h_m"])
        p = safe_float(c["p"])
        d = safe_float(c["d"])
        if not h_m or h_m <= 0.0:
            continue
        if not p or p < MIN_SWELL_PERIOD_S:
            continue
        p_eff = p * period_scale
        h_ft = m_to_ft(h_m)
        energy = round(h_ft ** 2 * p_eff, 1) if (h_ft and p_eff) else None
        comps.append({
            "height_ft":     h_ft,
            "period_s":      round(p_eff, 1) if p_eff else None,
            "direction_deg": d,
            "energy":        energy,
            "type":          c["type"],
        })

    comps.sort(key=lambda c: c["energy"] or 0, reverse=True)
    return comps if limit is None else comps[:limit]


WIND_SEA = "windsea"


def rank_with_wind_sea(swell_parts, wind_part, period_scale=1.0):
    """Rank the swell partitions together with the model's wind-sea partition.

    Both wave models file a sea under "wind waves" for as long as the local
    wind is still driving it, so during an onshore gale every swell partition
    reads 0 m while the combined sea is 10 ft @ 10 s (v1.13.4: GFS was empty
    in 55 % of the hours a buoy's primary read ≥ 8 ft @ ≥ 8 s). The buoy
    decomposition has no wind-sea concept — it reports that sea as its
    primary partition — so the models compete on the same terms: same 5 s
    floor, same H²·T ranking. wind_part=None keeps the ranking swell-only
    (/gland, whose window scoring must see both swell partitions).

    Returns (components, wind_sea, displaced_swell): the top 2; the wind-sea
    candidate wherever it ranked (None under the floor); and the swell it
    pushed out of the top 2, if any. The last two ride on the record so a
    swell-only ranking can be rebuilt from a logged row without a re-pull.
    """
    parts = list(swell_parts)
    if wind_part is not None:
        parts.append({**wind_part, "type": WIND_SEA})
    ranked = build_swell_components(parts, period_scale=period_scale, limit=None)
    comps = ranked[:2]
    wind_sea = next((c for c in ranked if c["type"] == WIND_SEA), None)
    displaced = None
    if wind_sea is not None and any(c is wind_sea for c in comps):
        swells = [c for c in ranked if c["type"] != WIND_SEA]
        displaced = swells[1] if len(swells) > 1 else None
    return comps, wind_sea, displaced


def make_wave_record(time_str, comps, primary, raw_direction,
                     combined_h_m, combined_p_s, combined_d_deg,
                     wind_sea=None, displaced_swell=None):
    """Assemble the canonical per-timestep record consumed by the frontend
    and the CSC2 forecast logger. This is the single source of the schema."""
    return {
        "time":               time_str,
        "wave_height_ft":     primary["height_ft"]     if primary else None,
        "wave_period_s":      primary["period_s"]      if primary else None,
        "wave_direction_deg": primary["direction_deg"] if primary else None,
        "energy":             primary["energy"]        if primary else None,
        "components":         comps,
        "raw_direction_deg":  raw_direction,
        # Combined (total) wave values — used by csc.predict, not the main UI.
        "combined_wave_height_m":      combined_h_m,
        "combined_wave_period_s":      combined_p_s,
        "combined_wave_direction_deg": combined_d_deg,
        # CSC2 only — see rank_with_wind_sea.
        "wind_sea":                    wind_sea,
        "displaced_swell":             displaced_swell,
    }
