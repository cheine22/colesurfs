"""CSC retrain notifier.

Called immediately after `python -m csc.train` succeeds on the seasonal
launchd schedule (com.colesurfs.csc-train.plist — equinoxes/solstices at
03:15). Does three things:

  1. Diffs the newest artifact's fold-averaged headline metrics against
     the currently-promoted artifact's (Hs MAE, Tp MAE, missed FUN+ days).
  2. Emits a 1-liner verdict through `send_notification` — the single
     place the transport layer lives. Default auto-selects Pushover (via
     the local pushover_notify.sh agent) if available, otherwise falls
     back to a macOS osascript banner.
  3. Writes a full Markdown report to /tmp/csc-train-report.md covering
     the headline delta + per-model fold-averaged metrics for review.

`python -m csc.notify --dry-run` prints the 1-liner + Markdown without
firing any notification.
`python -m csc.notify --test "msg"` sends a one-off notification through
the auto-selected channel and prints the result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from csc.schema import CSC_MODELS_DIR

REPORT_PATH = Path("/tmp/csc-train-report.md")

PUSHOVER_SCRIPT = Path(os.path.expanduser(
    "~/Documents/.config/pushover/pushover_notify.sh"))
PUSHOVER_ALERT_FILE = Path(os.path.expanduser(
    "~/Documents/.pushover_alert.json"))


# ─── Transport (pluggable) ───────────────────────────────────────────────

def _pushover_available() -> bool:
    """Pushover is available if the helper script exists and is executable."""
    try:
        return PUSHOVER_SCRIPT.exists() and os.access(PUSHOVER_SCRIPT, os.X_OK)
    except OSError:
        return False


def _send_pushover(title: str, body: str) -> bool:
    """Write the alert file the local pushover_notify.sh watches, then
    invoke it. Returns True on success."""
    try:
        PUSHOVER_ALERT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PUSHOVER_ALERT_FILE.write_text(json.dumps({
            "title": title,
            "message": body,
        }))
    except Exception:
        traceback.print_exc()
        return False
    try:
        res = subprocess.run(
            [str(PUSHOVER_SCRIPT)],
            check=False, timeout=20, capture_output=True,
        )
        # The script deletes the alert file on success; presence afterwards
        # indicates failure.
        if res.returncode != 0:
            return False
        return not PUSHOVER_ALERT_FILE.exists()
    except Exception:
        traceback.print_exc()
        return False


def _send_osascript(title: str, body: str) -> bool:
    safe_title = title.replace('"', "'").replace("\\", "/")
    safe_body = body.replace('"', "'").replace("\\", "/")
    script = (
        f'display notification "{safe_body}" '
        f'with title "{safe_title}" sound name "Submarine"'
    )
    try:
        res = subprocess.run(
            ["osascript", "-e", script],
            check=False, timeout=10, capture_output=True,
        )
        return res.returncode == 0
    except Exception:
        traceback.print_exc()
        return False


def send_notification(title: str, body: str, *, channel: str = "auto") -> dict:
    """Deliver a notification through the requested channel.

    channel:
      "auto"     — Pushover if available, else osascript
      "pushover" — Pushover only (no fallback)
      "osascript"— osascript only
      "both"     — fire both transports independently

    Returns a dict summarising which transports were attempted and their
    outcomes, e.g. {"channel": "auto", "pushover": True, "osascript": None}.
    A `None` entry means "not attempted".
    """
    result: dict[str, Any] = {"channel": channel,
                              "pushover": None, "osascript": None}

    ch = channel.lower()
    if ch not in ("auto", "pushover", "osascript", "both"):
        raise ValueError(f"unknown channel: {channel!r}")

    if ch == "osascript":
        result["osascript"] = _send_osascript(title, body)
        return result

    if ch == "pushover":
        result["pushover"] = _send_pushover(title, body)
        return result

    if ch == "both":
        result["pushover"] = _send_pushover(title, body)
        result["osascript"] = _send_osascript(title, body)
        return result

    # auto: Pushover first, osascript fallback on failure
    if _pushover_available():
        ok = _send_pushover(title, body)
        result["pushover"] = ok
        if ok:
            return result
    result["osascript"] = _send_osascript(title, body)
    return result


# ─── Artifact discovery ──────────────────────────────────────────────────

def _list_artifacts() -> list[Path]:
    root = Path(CSC_MODELS_DIR)
    if not root.exists():
        return []
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.is_symlink():
            continue
        if d.name.startswith(".") or d.name == "current":
            continue
        if (d / "manifest.json").exists() and (d / "metrics.json").exists():
            out.append(d)
    return out


def _newest_artifact() -> Path | None:
    arts = _list_artifacts()
    if not arts:
        return None
    # sort by mtime of manifest.json so the post-train newcomer wins
    arts.sort(key=lambda p: (p / "manifest.json").stat().st_mtime)
    return arts[-1]


def _current_artifact() -> Path | None:
    link = Path(CSC_MODELS_DIR) / "current"
    if not link.exists():
        return None
    try:
        return link.resolve()
    except OSError:
        return None


def _load_manifest(artifact_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((artifact_dir / "manifest.json").read_text())
    except Exception:
        return None


def _load_metrics(artifact_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((artifact_dir / "metrics.json").read_text())
    except Exception:
        return None


# ─── Headline extraction ─────────────────────────────────────────────────

def _winner_block(metrics: dict[str, Any], winner: str | None) -> dict[str, Any]:
    """Pull the winner model's fold-averaged top-level stats out of
    metrics.json (written by csc.experiment.run_bakeoff)."""
    if not metrics or "metrics" not in metrics:
        return {}
    by_model = metrics["metrics"]
    if winner and winner in by_model:
        return by_model[winner]
    # fallback: any key that has hs_mae
    for _k, v in by_model.items():
        if isinstance(v, dict) and "hs_mae" in v:
            return v
    return {}


def _missed_fun_days(artifact_dir: Path) -> int | None:
    """Read funplus detection counts if present — `misses` in the fun_plus
    detection contingency table is the missed-FUN+ hour count. We surface
    it unchanged (per-hour) — the naming is historical."""
    fp = artifact_dir / "fun_plus_report.json"
    if not fp.exists():
        return None
    try:
        j = json.loads(fp.read_text())
    except Exception:
        return None
    # Prefer the winner model's entry; otherwise any model with a
    # fun_plus_detection block.
    models = j.get("models") or {}
    # try funplus first, then lgbm, then any
    for cand in ("funplus", "lgbm", "ridge_mos", "lgbm_per_coast"):
        if cand in models:
            det = models[cand].get("fun_plus_detection")
            if det and "misses" in det:
                return int(det["misses"])
    for _k, v in models.items():
        det = (v or {}).get("fun_plus_detection")
        if det and "misses" in det:
            return int(det["misses"])
    return None


def _fmt_pct(old: float, new: float) -> str:
    if old is None or new is None or old == 0 or math.isnan(old) or math.isnan(new):
        return "n/a"
    pct = (new - old) / old * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _fmt_num(x, places: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{places}f}"


def headline(new_dir: Path, cur_dir: Path | None) -> dict[str, Any]:
    """Compute the 1-line summary deltas. Returns a dict for structured
    logging + a formatted `summary` string."""
    new_manifest = _load_manifest(new_dir) or {}
    new_metrics = _load_metrics(new_dir) or {}
    new_winner = new_manifest.get("winner")
    new_block = _winner_block(new_metrics, new_winner)

    cur_manifest = _load_manifest(cur_dir) if cur_dir else None
    cur_metrics = _load_metrics(cur_dir) if cur_dir else None
    cur_winner = (cur_manifest or {}).get("winner")
    cur_block = _winner_block(cur_metrics or {}, cur_winner)

    new_hs = new_block.get("hs_mae")
    cur_hs = cur_block.get("hs_mae")
    new_tp = new_block.get("tp_mae")
    cur_tp = cur_block.get("tp_mae")
    new_miss = _missed_fun_days(new_dir)
    cur_miss = _missed_fun_days(cur_dir) if cur_dir else None

    version_tag = new_manifest.get("version") or new_dir.name

    # Scoring: strictly-better-than-current on Hs MAE triggers the
    # "worth considering promote" verdict.
    better_hs = (
        new_hs is not None and cur_hs is not None
        and new_hs < cur_hs
    )

    parts: list[str] = []
    if new_hs is not None and cur_hs is not None:
        parts.append(
            f"Hs MAE {cur_hs:.3f} → {new_hs:.3f} m ({_fmt_pct(cur_hs, new_hs)})"
        )
    elif new_hs is not None:
        parts.append(f"Hs MAE {new_hs:.3f} m (no prior)")
    if new_miss is not None and cur_miss is not None:
        parts.append(
            f"missed FUN+ {cur_miss} → {new_miss} ({_fmt_pct(cur_miss, new_miss)})"
        )
    elif new_miss is not None:
        parts.append(f"missed FUN+ {new_miss} (no prior)")

    verdict = "Worth considering promote." if better_hs else "Keep current."
    summary = f"New CSC {version_tag}: " + "; ".join(parts) + f" {verdict}"

    return {
        "version": version_tag,
        "new_dir": str(new_dir),
        "current_dir": str(cur_dir) if cur_dir else None,
        "new": {"hs_mae": new_hs, "tp_mae": new_tp, "missed_fun": new_miss,
                "winner": new_winner},
        "current": {"hs_mae": cur_hs, "tp_mae": cur_tp, "missed_fun": cur_miss,
                    "winner": cur_winner},
        "better_hs": better_hs,
        "summary": summary,
    }


# ─── Markdown report ─────────────────────────────────────────────────────

def _model_table(label: str, metrics: dict[str, Any]) -> str:
    """Render a markdown table of fold-averaged metrics per model."""
    if not metrics or "metrics" not in metrics:
        return f"### {label}\n\n_no metrics.json_\n"
    rows = ["| model | hs_mae | tp_mae | dp_circ_mae | hs_rmse | tp_rmse |",
            "|---|---|---|---|---|---|"]
    for name, block in metrics["metrics"].items():
        if not isinstance(block, dict):
            continue
        rows.append(
            f"| `{name}` | {_fmt_num(block.get('hs_mae'))} | "
            f"{_fmt_num(block.get('tp_mae'))} | "
            f"{_fmt_num(block.get('dp_circ_mae'), 2)} | "
            f"{_fmt_num(block.get('hs_rmse'))} | "
            f"{_fmt_num(block.get('tp_rmse'))} |"
        )
    return f"### {label}\n\n" + "\n".join(rows) + "\n"


def build_report(hd: dict[str, Any], new_dir: Path,
                 cur_dir: Path | None) -> str:
    new_metrics = _load_metrics(new_dir) or {}
    cur_metrics = _load_metrics(cur_dir) if cur_dir else {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# CSC retrain — {hd['version']}",
        "",
        f"_Generated {now}_",
        "",
        "## Headline",
        "",
        f"> {hd['summary']}",
        "",
        "## Winner comparison",
        "",
        f"- **New** winner: `{hd['new'].get('winner')}` "
        f"(dir: `{new_dir}`)",
        f"- **Current** winner: `{hd['current'].get('winner')}` "
        f"(dir: `{cur_dir}`)",
        "",
        "| metric | current | new | Δ |",
        "|---|---|---|---|",
        f"| Hs MAE (m) | {_fmt_num(hd['current'].get('hs_mae'))} | "
        f"{_fmt_num(hd['new'].get('hs_mae'))} | "
        f"{_fmt_pct(hd['current'].get('hs_mae'), hd['new'].get('hs_mae'))} |",
        f"| Tp MAE (s) | {_fmt_num(hd['current'].get('tp_mae'))} | "
        f"{_fmt_num(hd['new'].get('tp_mae'))} | "
        f"{_fmt_pct(hd['current'].get('tp_mae'), hd['new'].get('tp_mae'))} |",
        f"| missed FUN+ | {hd['current'].get('missed_fun')} | "
        f"{hd['new'].get('missed_fun')} | "
        f"{_fmt_pct(hd['current'].get('missed_fun'), hd['new'].get('missed_fun'))} |",
        "",
        "## Per-model fold-averaged metrics",
        "",
        _model_table("New artifact", new_metrics),
        "",
        _model_table("Current (promoted) artifact", cur_metrics),
        "",
        "## To promote",
        "",
        f"```sh",
        f"python -m csc.promote {shlex.quote(str(new_dir))}",
        f"```",
        "",
    ]
    return "\n".join(lines)


# ─── CLI entry ───────────────────────────────────────────────────────────

def _resolve_dirs(args) -> tuple[Path | None, Path | None]:
    new_dir = Path(args.new) if args.new else _newest_artifact()
    cur_dir = Path(args.current) if args.current else _current_artifact()
    return new_dir, cur_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csc.notify")
    ap.add_argument("--new", default=None,
                    help="Path to the new artifact (default: newest by mtime).")
    ap.add_argument("--current", default=None,
                    help="Path to the current artifact "
                         "(default: .csc_models/current).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the 1-liner + Markdown without firing "
                         "a notification.")
    ap.add_argument("--test", default=None, metavar="MSG",
                    help="Send a one-off test notification through the "
                         "auto-selected channel and exit.")
    ap.add_argument("--channel", default="auto",
                    choices=("auto", "pushover", "osascript", "both"),
                    help="Transport channel (default: auto).")
    args = ap.parse_args(argv)

    if args.test is not None:
        result = send_notification("CSC notify test", args.test,
                                   channel=args.channel)
        print(f"[notify] channel={args.channel} pushover_available="
              f"{_pushover_available()} result={result}")
        # Success if any transport returned True
        ok = any(v is True for v in
                 (result.get("pushover"), result.get("osascript")))
        return 0 if ok else 1

    new_dir, cur_dir = _resolve_dirs(args)
    if new_dir is None:
        print("[notify] no new artifact found — nothing to do",
              file=sys.stderr)
        return 1

    hd = headline(new_dir, cur_dir)
    report = build_report(hd, new_dir, cur_dir)

    try:
        REPORT_PATH.write_text(report)
        print(f"[notify] wrote {REPORT_PATH}")
    except Exception:
        traceback.print_exc()

    print(hd["summary"])
    if args.dry_run:
        print("\n--- /tmp/csc-train-report.md ---\n")
        print(report)
        return 0

    title_stamp = hd["version"]
    result = send_notification(f"CSC retrain {title_stamp}", hd["summary"],
                               channel=args.channel)
    print(f"[notify] sent via {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
