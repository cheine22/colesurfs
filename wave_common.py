"""
colesurfs — shared wave-record processing used by waves.py (Open-Meteo GFS)
and waves_cmems.py (CMEMS EURO).

Extracted in the v1.9 consolidation. Behavior is locked by
tests/test_wave_identity.py — any change here must be an intentional,
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


def build_swell_components(raw_parts, period_scale=1.0):
    """Filter + shape raw swell partitions into the display component list.

    raw_parts: iterable of {"h_m", "p", "d", "type"} dicts, in partition order.
    period_scale: multiplier applied to the raw period before display/energy
    (CMEMS passes 1.20 to convert Tm01 → Tp; Open-Meteo periods are already Tp).

    Returns the top-2 components by energy (h_ft² × Tp).
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
    return comps[:2]


def make_wave_record(time_str, comps, primary, raw_direction,
                     combined_h_m, combined_p_s, combined_d_deg):
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
    }
