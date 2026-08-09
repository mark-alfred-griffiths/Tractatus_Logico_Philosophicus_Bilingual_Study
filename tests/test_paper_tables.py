from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "paper" / "tables"
MANIFEST = TABLE_DIR / "table_manifest.csv"


class PaperTableManifestTests(unittest.TestCase):
    def test_table_manifest_paths_exist(self) -> None:
        self.assertTrue(MANIFEST.is_file())
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        for row in rows:
            with self.subTest(table_id=row["table_id"]):
                table_file = ROOT / row["table_file"]
                source_data = ROOT / row["source_data"]
                self.assertTrue(table_file.is_file(), table_file)
                self.assertTrue(source_data.is_file(), source_data)
                self.assertEqual(row["canonical_status"], "derived_paper_output")
                self.assertTrue(row["generation_script"])
                self.assertTrue(row["verification_command"])

    def test_table_exports_match_declared_sources(self) -> None:
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        for row in rows:
            with self.subTest(table_id=row["table_id"]):
                table_file = ROOT / row["table_file"]
                source_data = ROOT / row["source_data"]
                self.assertEqual(
                    table_file.read_bytes(),
                    source_data.read_bytes(),
                    f"{table_file} differs from {source_data}",
                )


if __name__ == "__main__":
    unittest.main()
