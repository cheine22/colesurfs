"""CSC2 model registry — discovers trained models on disk, scores them
against a fixed reference (raw EURO holdout MAE), and selects the
top-3 set that surfaces on `/csc`.

Selection rule (per spec): always show the **#1 performing model** plus
**the 2 most recent additional models** (different by name from #1).
This keeps the leaderboard's best entry pinned while still surfacing
fresh experiments. If only 1 or 2 models exist, returns what's there
without padding.

Composite skill score:
  skill[v]    = 1 - (model_MAE[v] / raw_EURO_MAE[v])    per variable
  composite   = mean(skill[v]) across the 6 dashboard swell variables
                (sw1/sw2 × {height_ft, period_s, direction_deg})

  composite > 0  → beats raw EURO on average
  composite = 0  → matches EURO
  composite < 0  → worse than EURO

Direction MAE is in degrees (circular MAE) and counts on equal footing;
EURO direction MAE acts as the natural unit for that variable's skill.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from csc2.schema import CSC2_MODELS_DIR

VARIABLES = (
    "sw1_height_ft", "sw1_period_s", "sw1_direction_deg",
    "sw2_height_ft", "sw2_period_s", "sw2_direction_deg",
)

# Composite-skill weights. Primary swell height + period and the surfer
# FUN_OR_BETTER F1 are the components surfers care about most, so they
# carry the lion's share. Direction and the secondary swell partitions
# round out the score with light weight.
SKILL_WEIGHTS = {
    "sw1_height_ft":     0.25,
    "sw1_period_s":      0.25,
    "sw1_direction_deg": 0.05,
    "sw2_height_ft":     0.05,
    "sw2_period_s":      0.05,
    "sw2_direction_deg": 0.05,
    "surfer_F1":         0.30,
}
assert abs(sum(SKILL_WEIGHTS.values()) - 1.0) < 1e-9

# A model is eligible for the "#1 performer" slot only if its holdout has
# at least this many rows. Skill scores from very small holdouts are too
# noisy to rank against models with year-scale test sets. Models below
# the threshold can still appear via the "most recent" slots.
MIN_TEST_ROWS_FOR_BEST = 1000


def _parse_yymmdd(name: str) -> str:
    """Pull the YYMMDD chunk from a model name (between two underscores)."""
    parts = name.split("_")
    if len(parts) >= 2 and len(parts[1]) == 6 and parts[1].isdigit():
        return parts[1]
    return ""


def _trained_utc_or_zero(meta: dict) -> float:
    s = meta.get("trained_utc")
    if not s:
        return 0.0
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except Exception:
        return 0.0


def _model_metric_block(meta: dict) -> dict:
    """Pull the model's own metric block from meta.json (keyed by 'baseline'
    or 'ml' depending on architecture). Returns {} if missing."""
    metrics = meta.get("metrics") or {}
    for key in ("ml", "baseline", "model"):
        if key in metrics:
            return metrics[key]
    return {}


def _ref_metric_block(meta: dict) -> dict:
    """raw_EURO is the reference for skill normalization."""
    return (meta.get("metrics") or {}).get("raw_EURO", {})


def _f1(sens, ppv):
    """F1 = 2·sens·ppv / (sens + ppv). Returns None if either is missing or both 0."""
    if sens is None or ppv is None:
        return None
    if (sens + ppv) <= 0:
        return 0.0
    return 2.0 * sens * ppv / (sens + ppv)


def composite_skill(meta: dict) -> float | None:
    """Weighted skill score across the 6 dashboard variables (MAE-based)
    plus surfer FUN_OR_BETTER F1. Returns None if any required component
    is missing on either the model or the raw_EURO reference, so we never
    compare half-populated leaderboards.

    Component skill normalization:
      - MAE vars: skill = 1 − (model_MAE / euro_MAE). 0 means matches EURO,
        +1 means perfect, negative means worse.
      - Surfer F1: skill = (model_F1 − euro_F1) / max(0.01, 1 − euro_F1).
        Same semantics: 0 means matches EURO, +1 means perfect.
    """
    model_b = _model_metric_block(meta)
    ref_b   = _ref_metric_block(meta)
    if not model_b or not ref_b:
        return None

    components: dict[str, float] = {}

    for v in VARIABLES:
        m = (model_b.get(v) or {}).get("mae")
        r = (ref_b.get(v)   or {}).get("mae")
        if m is None or r is None or r <= 0:
            return None
        components[v] = 1.0 - (m / r)

    # Surfer F1: pulled from FUN_OR_BETTER block. If the model lacks
    # surfer metrics, return None (surfer carries 30% — too much to skip).
    m_surf = (model_b.get("surfer") or {}).get("FUN_OR_BETTER") or {}
    r_surf = (ref_b.get("surfer")   or {}).get("FUN_OR_BETTER") or {}
    m_f1 = _f1(m_surf.get("sens"), m_surf.get("ppv"))
    r_f1 = _f1(r_surf.get("sens"), r_surf.get("ppv"))
    if m_f1 is None or r_f1 is None:
        return None
    components["surfer_F1"] = (m_f1 - r_f1) / max(0.01, 1.0 - r_f1)

    return sum(components[k] * w for k, w in SKILL_WEIGHTS.items())


def _arch(name: str) -> str:
    """'CSC2+baseline_…' → 'baseline', 'CSC2+ML_…' → 'ML'."""
    if name.startswith("CSC2+baseline"):
        return "baseline"
    if name.startswith("CSC2+ML"):
        return "ML"
    return "?"


def list_models(scope: str = "east") -> list[dict]:
    """Discover trained models in `.csc2_models/<scope>/`. Each entry:
        {name, dir, arch, trained_utc, trained_ts, date_yymmdd, version,
         coverage_frac, composite_skill, metrics, meta}
    Sorted by trained_utc descending (newest first)."""
    root = CSC2_MODELS_DIR / scope
    if not root.exists():
        return []
    out: list[dict] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("CSC2+"):
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        skill = composite_skill(meta)
        # version suffix is the last underscore-separated chunk if it starts
        # with 'v' followed by digits.
        ver = ""
        last = d.name.rsplit("_", 1)[-1]
        if last.startswith("v") and last[1:].isdigit():
            ver = last
        out.append({
            "name":        d.name,
            "dir":         str(d),
            "arch":        _arch(d.name),
            "trained_utc": meta.get("trained_utc"),
            "trained_ts":  _trained_utc_or_zero(meta),
            "date_yymmdd": _parse_yymmdd(d.name),
            "version":     ver,
            "coverage_frac": meta.get("coverage_frac"),
            "composite_skill": skill,
            "metrics":     meta.get("metrics") or {},
            "meta":        meta,
        })
    out.sort(key=lambda x: x["trained_ts"], reverse=True)
    return out


def select_top3(models: Iterable[dict]) -> list[dict]:
    """Return [#1 by skill, 2 most recent that aren't #1].

    If there's a tie at #1 (same composite skill), the more recently
    trained one wins — that's a natural tiebreaker. The "recent 2" are
    drawn from the trained_utc-descending list, skipping #1 if encountered.
    Returns whatever subset exists when fewer than 3 models are scored.
    """
    pool = [
        m for m in models
        if m.get("composite_skill") is not None
        and (m["meta"].get("n_test_rows") or 0) >= MIN_TEST_ROWS_FOR_BEST
    ]
    if not pool:
        # No scored models with adequate holdout — fall back to recency
        # only, which is more honest than ranking 420-row holdouts.
        return list(models)[:3]
    # Best by skill (recency tiebreaker).
    best = max(pool, key=lambda m: (m["composite_skill"], m["trained_ts"]))
    # Most recent two that aren't best.
    recent = [m for m in models if m["name"] != best["name"]][:2]
    return [best] + recent


def selection_payload(scope: str = "east") -> dict:
    """JSON-serializable structure for the /api/csc2/models endpoint."""
    all_models = list_models(scope)
    selected = select_top3(all_models)
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": scope,
        "total_models": len(all_models),
        "selected": [
            {
                "name":            m["name"],
                "arch":            m["arch"],
                "trained_utc":     m["trained_utc"],
                "date_yymmdd":     m["date_yymmdd"],
                "version":         m["version"],
                "coverage_frac":   m["coverage_frac"],
                "composite_skill": (round(m["composite_skill"], 4)
                                    if m["composite_skill"] is not None else None),
                "is_top_performer": (i == 0),
                "metrics":         m["metrics"],
                "n_train_rows":    m["meta"].get("n_train_rows"),
                "n_test_rows":     m["meta"].get("n_test_rows"),
            }
            for i, m in enumerate(selected)
        ],
        "all_models": [
            {
                "name":            m["name"],
                "arch":            m["arch"],
                "trained_utc":     m["trained_utc"],
                "composite_skill": (round(m["composite_skill"], 4)
                                    if m["composite_skill"] is not None else None),
            }
            for m in all_models
        ],
    }


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(selection_payload(), indent=2, default=str))
