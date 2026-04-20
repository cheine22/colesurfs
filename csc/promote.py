"""Atomically point .csc_models/current at a specific trained version."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from csc.schema import CSC_MODELS_DIR


def promote(target: Path) -> int:
    target = target.resolve()
    if not target.is_dir():
        print(f"[promote] not a directory: {target}", file=sys.stderr)
        return 2
    if not (target / "manifest.json").exists():
        print(f"[promote] missing manifest.json in {target}", file=sys.stderr)
        return 2
    link = CSC_MODELS_DIR / "current"
    CSC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic relink: make new, rename over old
    tmp = CSC_MODELS_DIR / ".current.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link)
    print(f"[promote] current → {os.readlink(link)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="csc.promote")
    ap.add_argument("target", type=Path,
                    help="Directory under .csc_models/ to promote.")
    args = ap.parse_args()
    return promote(args.target)


if __name__ == "__main__":
    sys.exit(main())
