#!/usr/bin/env python3
"""Validate the final-paper figure manifest without regenerating figures."""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_MANIFEST = Path("paper/figures/figure_manifest.csv")


def iter_paths(value: str) -> list[Path]:
    if not value.strip():
        return []
    return [Path(part.strip()) for part in value.split(";") if part.strip()]


def png_dpi(path: Path) -> tuple[float, float] | None:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        return None

    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if chunk_type == b"pHYs":
            x_ppm, y_ppm, unit = struct.unpack(">IIB", chunk)
            if unit != 1:
                return None
            return x_ppm * 0.0254, y_ppm * 0.0254
        offset += 12 + length
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--required-dpi", type=float, default=600.0)
    parser.add_argument("--dpi-tolerance", type=float, default=0.5)
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"missing manifest: {args.manifest}", file=sys.stderr)
        return 1

    failures: list[str] = []
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    listed_files: set[Path] = {args.manifest}
    for line_number, row in enumerate(rows, start=2):
        if row.get("used_in_final_manuscript") != "yes":
            failures.append(
                f"line {line_number}: non-paper figure row is not allowed in final manifest: "
                f"{row.get('figure_id', '<unknown>')}"
            )

        for column in ("figure_file", "companion_file", "source_data"):
            for path in iter_paths(row.get(column, "")):
                listed_files.add(path)
                if not path.is_file():
                    failures.append(f"line {line_number}: missing {column}: {path}")

        if not row.get("sha256", "").strip():
            failures.append(f"line {line_number}: empty sha256")

        if row.get("used_in_final_manuscript") == "yes":
            for path in iter_paths(row.get("figure_file", "")):
                if path.suffix.lower() != ".png":
                    continue
                dpi = png_dpi(path)
                if dpi is None:
                    failures.append(f"line {line_number}: PNG has no dpi metadata: {path}")
                    continue
                x_dpi, y_dpi = dpi
                if (
                    abs(x_dpi - args.required_dpi) > args.dpi_tolerance
                    or abs(y_dpi - args.required_dpi) > args.dpi_tolerance
                ):
                    failures.append(
                        f"line {line_number}: {path} is {x_dpi:.3f}x{y_dpi:.3f} dpi"
                    )

    figure_dir = args.manifest.parent
    if figure_dir.is_dir():
        for path in sorted(figure_dir.iterdir()):
            if not path.is_file():
                continue
            if path not in listed_files:
                failures.append(f"unlisted file in final paper figure directory: {path}")

    if failures:
        print("figure manifest validation failed", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"validated {len(rows)} figure manifest rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
