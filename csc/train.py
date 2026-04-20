"""CSC train — wraps csc.experiment, picks the winner, writes the
canonical artifact that csc.serve loads at inference.

By design, this does NOT flip .csc_models/current — promotion is manual
via `python -m csc.promote <dir>` per the approved plan's notify-then-
promote flow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from csc.experiment import run_bakeoff, run_bakeoff_primary
from csc.schema import CSC_MODELS_DIR, models_dir_for_scope

try:
    from csc.funplus import evaluate_artifact as _funplus_evaluate
    from csc.funplus import _print_console_report as _funplus_print
    from csc.funplus import _verdict as _funplus_verdict
except Exception:  # keep train resilient even if funplus import fails
    _funplus_evaluate = None
    _funplus_print = None
    _funplus_verdict = None


def _pick_winner(metrics: dict[str, dict]) -> str:
    """Winner = the trained *global-scope* model with the best CV-averaged
    Hs MAE. Per-coast specialists (`lgbm_east`/`lgbm_west`) are excluded
    because their MAE is computed on only half the test set and isn't
    directly comparable. Falls back to 'mean' if nothing trained."""
    global_candidates = ("ridge_mos", "lgbm", "funplus", "lgbm_per_coast")
    candidates = [k for k in global_candidates
                  if k in metrics and "error" not in metrics[k]]
    scored = []
    for k in candidates:
        v = metrics[k].get("hs_mae")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        scored.append((v, k))
    if not scored:
        return "mean"
    scored.sort()
    return scored[0][1]


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.train")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--primary", action="store_true",
                    help="v2+ retrain targeting NDBC primary-swell Hs "
                         "(recommended default). Uses "
                         "build_training_frame_primary(). Does NOT "
                         "auto-promote — same notify-then-human-promote flow.")
    ap.add_argument("--scope", choices=("east", "west"), default="east",
                    help="v3 scope separation: train on East OR West Coast "
                         "buoys independently. East artifacts land in "
                         ".csc_models/ (dashboard default); West goes to "
                         ".csc_models_west/ (silent, never on /csc). "
                         "Default: east. Ignored when --primary is not set.")
    ap.add_argument("--quiet-notify", action="store_true",
                    help="Suppress Pushover/osascript notification on "
                         "completion. West Coast training runs with this "
                         "on so it stays off the user's radar.")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    if args.primary:
        root = models_dir_for_scope(args.scope)
        out_dir = args.out or (root / f"{stamp}_{args.scope}_v3")
    else:
        out_dir = args.out or (CSC_MODELS_DIR / f"{stamp}_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.primary:
        report = run_bakeoff_primary(out_dir, scope=args.scope)
    else:
        report = run_bakeoff(out_dir)
    winner = _pick_winner(report["metrics"])
    print(f"[train] picked winner: {winner}")

    manifest = {
        "version": stamp,
        "winner": winner,
        "winner_dir": str((out_dir / winner).resolve()),
        "scope": args.scope if args.primary else "all",
        "generated_at": report["generated_at"],
        "train_rows": report["train_rows"],
        "test_rows": report["test_rows"],
        "holdout_cutoff": report["holdout_cutoff"],
        "buoys": report["buoys"],
        "cv": report.get("cv", {}),
        "target": report.get("target", "combined"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[train] wrote manifest → {out_dir / 'manifest.json'}")
    if args.primary:
        promote_target = out_dir.parent / "current"
        print(f"[train] to promote ({args.scope} v3):  "
              f"ln -sfn {out_dir.name} {promote_target}")
    else:
        print(f"[train] to promote:  python -m csc.promote {out_dir}")

    # FUN+-specific side report — compares funplus vs the other candidates
    # on the surfable-swell subset only. Purely informational; does not
    # affect winner selection.
    if _funplus_evaluate is not None:
        try:
            fp_report = _funplus_evaluate(out_dir)
            _funplus_print(fp_report)
            print(_funplus_verdict(fp_report))
        except Exception as e:
            print(f"[train] funplus post-eval skipped: {e!r}")

    # Notify (same flow as v1 seasonal train): does NOT auto-promote.
    # West Coast training is always silent — user doesn't see West reports
    # unless explicitly requested.
    suppress_notify = args.quiet_notify or args.scope == "west"
    if args.primary and not suppress_notify:
        try:
            from csc import notify as csc_notify
            csc_notify.main([])
        except Exception as e:
            print(f"[train] notify skipped: {e!r}")
    elif suppress_notify:
        print(f"[train] ({args.scope}) notifications suppressed — artifact written silently")
    return 0


if __name__ == "__main__":
    sys.exit(main())
