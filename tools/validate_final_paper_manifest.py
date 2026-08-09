#!/usr/bin/env python3
"""Validate paths and hashes in the final-paper artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "paper" / "final_paper_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    if not manifest.is_file():
        print(f"missing manifest: {manifest}", file=sys.stderr)
        return 1

    failures: list[str] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for line_number, row in enumerate(rows, start=2):
        rel_path = row.get("path", "").strip()
        expected_hash = row.get("sha256", "").strip()
        if not rel_path:
            failures.append(f"line {line_number}: empty path")
            continue
        path = ROOT / rel_path
        if not path.is_file():
            failures.append(f"line {line_number}: missing path: {rel_path}")
            continue
        if not expected_hash:
            failures.append(f"line {line_number}: empty sha256: {rel_path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            failures.append(f"line {line_number}: sha256 mismatch: {rel_path}")

    if failures:
        print("final-paper manifest validation failed", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"validated {len(rows)} final-paper manifest rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
