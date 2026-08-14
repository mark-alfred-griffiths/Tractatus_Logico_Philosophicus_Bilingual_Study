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
DEFAULT_VALIDATION_DIR = Path("paper/figures/validation")
EXPECTED_VALIDATION_FILES = {
    "Figure_1_Retained_Ablation_Diagnostics_publication.pdf",
    "Figure_1_Retained_Ablation_Diagnostics_publication.png",
    "Figure_1_Retained_Ablation_Diagnostics_publication.tiff",
    "Figure_2_Retained_vs_Holdout_Retrieval_publication.pdf",
    "Figure_2_Retained_vs_Holdout_Retrieval_publication.png",
    "Figure_2_Retained_vs_Holdout_Retrieval_publication.tiff",
    "figure_1_plot_data.csv",
    "figure_2_plot_data.csv",
}


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


def _tiff_scalar(data: bytes, endian: str, field_type: int, count: int, value_offset: bytes) -> int | None:
    if field_type == 3 and count == 1:
        return struct.unpack(endian + "H", value_offset[:2])[0]
    if field_type == 4 and count == 1:
        return struct.unpack(endian + "I", value_offset)[0]
    return None


def _tiff_rational(data: bytes, endian: str, field_type: int, count: int, value_offset: bytes) -> float | None:
    if field_type != 5 or count != 1:
        return None
    offset = struct.unpack(endian + "I", value_offset)[0]
    if offset + 8 > len(data):
        return None
    numerator, denominator = struct.unpack(endian + "II", data[offset : offset + 8])
    if denominator == 0:
        return None
    return numerator / denominator


def tiff_dpi(path: Path) -> tuple[float, float] | None:
    data = path.read_bytes()
    if len(data) < 8:
        return None
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return None
    if struct.unpack(endian + "H", data[2:4])[0] != 42:
        return None
    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    if ifd_offset + 2 > len(data):
        return None
    entry_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    x_resolution: float | None = None
    y_resolution: float | None = None
    resolution_unit = 2
    for entry_i in range(entry_count):
        start = ifd_offset + 2 + entry_i * 12
        if start + 12 > len(data):
            return None
        tag, field_type, count = struct.unpack(endian + "HHI", data[start : start + 8])
        value_offset = data[start + 8 : start + 12]
        if tag == 282:
            x_resolution = _tiff_rational(data, endian, field_type, count, value_offset)
        elif tag == 283:
            y_resolution = _tiff_rational(data, endian, field_type, count, value_offset)
        elif tag == 296:
            unit = _tiff_scalar(data, endian, field_type, count, value_offset)
            if unit is not None:
                resolution_unit = unit
    if x_resolution is None or y_resolution is None:
        return None
    if resolution_unit == 2:
        return x_resolution, y_resolution
    if resolution_unit == 3:
        return x_resolution * 2.54, y_resolution * 2.54
    return None


def raster_dpi(path: Path) -> tuple[float, float] | None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return png_dpi(path)
    if suffix in {".tif", ".tiff"}:
        return tiff_dpi(path)
    return None


def check_raster_dpi(path: Path, required_dpi: float, dpi_tolerance: float, failures: list[str], label: str) -> None:
    dpi = raster_dpi(path)
    if dpi is None:
        failures.append(f"{label} has no dpi metadata: {path}")
        return
    x_dpi, y_dpi = dpi
    if abs(x_dpi - required_dpi) > dpi_tolerance or abs(y_dpi - required_dpi) > dpi_tolerance:
        failures.append(f"{label} is {x_dpi:.3f}x{y_dpi:.3f} dpi: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--require-validation", action="store_true")
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
                if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                    continue
                check_raster_dpi(path, args.required_dpi, args.dpi_tolerance, failures, f"line {line_number}: raster")

    figure_dir = args.manifest.parent
    if figure_dir.is_dir():
        for path in sorted(figure_dir.iterdir()):
            if not path.is_file():
                continue
            if path not in listed_files:
                failures.append(f"unlisted file in final paper figure directory: {path}")

    if args.require_validation or args.validation_dir.exists():
        if not args.validation_dir.is_dir():
            failures.append(f"missing validation figure directory: {args.validation_dir}")
        else:
            found = {path.name for path in args.validation_dir.iterdir() if path.is_file()}
            missing = sorted(EXPECTED_VALIDATION_FILES - found)
            extra = sorted(found - EXPECTED_VALIDATION_FILES)
            for name in missing:
                failures.append(f"missing validation figure artifact: {args.validation_dir / name}")
            for name in extra:
                failures.append(f"unlisted file in validation figure directory: {args.validation_dir / name}")
            for name in sorted(found & EXPECTED_VALIDATION_FILES):
                path = args.validation_dir / name
                if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                    continue
                check_raster_dpi(path, args.required_dpi, args.dpi_tolerance, failures, "validation raster")

    if failures:
        print("figure manifest validation failed", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"validated {len(rows)} figure manifest rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
