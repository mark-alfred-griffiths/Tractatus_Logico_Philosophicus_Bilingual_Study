# Full Reproducibility Test Plan

This document gives an ordered test sequence for a fresh clone of the final
paper companion repository. It separates fast non-training checks, external
archive restoration, dataset rebuilding, and full empirical reruns.

Run commands from the repository root unless stated otherwise.

## 0. Create A Clean Environment

Clone the repository into a new directory and create a fresh Conda
environment:

```bash
git clone git@github.com:mark-alfred-griffiths/Tractatus_Logico_Philosophicus_Bilingual_Study.git
cd Tractatus_Logico_Philosophicus_Bilingual_Study
conda create -n tractatus-repro python=3.10
conda activate tractatus-repro
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Expected result: the `tractatus-repro` Conda environment activates and
dependencies install without errors.

## 1. Verify Clean-Clone Small Evidence

These checks should pass without downloading the heavy archive and without
training models:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
python3 tools/validate_final_paper_manifest.py
python3 tools/validate_paper_figure_manifest.py --require-validation
python3 -m pytest
```

Expected result:

- the dry run prints commands only
- the final-paper manifest validates
- the figure manifest validates and confirms the retained PNG has 600 dpi
- pytest passes

If this stage fails, the GitHub repository is missing a small tracked artifact
or has an inconsistent manifest.

## 2. Regenerate Paper-Facing Derivatives Without Training

Run the safe wrapper:

```bash
python3 tools/reproduce_final_paper_outputs.py
```

Expected result: canonical reports, paper tables, figure manifest checks, and
the final-paper manifest are regenerated or verified from tracked retained
evidence. This command must not create checkpoints or raw parquet outputs.

Confirm the working tree:

```bash
git status --short --ignored
```

Expected result: no unexpected tracked changes. Ignored local cache files may
appear.

## 3. Restore Heavy External Artifacts

Download the heavy archive from the GitHub Release or external archive record.
The expected files are:

```text
tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
tractatus-heavy-retained-artifacts-2026-08-09.tar.gz.sha256
```

Verify the archive hash:

```bash
sha256sum tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
cat tractatus-heavy-retained-artifacts-2026-08-09.tar.gz.sha256
```

Expected SHA256:

```text
731891afbe1ebc300c5f3610838b943994745c51167b28917e0562a9fb4bf576
```

Restore the archive into the repository root:

```bash
tar -xzf tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
```

Validate restored heavy files against the tracked manifest:

```bash
python3 - <<'PY'
import csv
import hashlib
from pathlib import Path

root = Path(".")
failures = []
with Path("docs/heavy_artifacts_manifest.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        path = root / row["path"]
        if not path.is_file():
            failures.append(f"missing {row['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            failures.append(f"sha256 mismatch {row['path']}")
if failures:
    raise SystemExit("\n".join(failures))
print("heavy artifact manifest validated")
PY
```

Expected result: every archived file exists and matches the SHA256 recorded in
`docs/heavy_artifacts_manifest.csv`.

## 4. Verify Restored Heavy Artifacts

The validation bundle is archival packaging and is not required for canonical
paper reproducibility. Do not run `--include-bundle` as part of the standard
test plan.

After restoring the heavy archive, verify the canonical evidence layer:

```bash
python3 tools/verify_canonical_evidence.py
python3 tools/reproduce_final_paper_outputs.py
python3 -m pytest
```

Expected result:

- restored checkpoints, raw parquet outputs, and smoke outputs remain local
- canonical evidence verification passes
- paper-facing derived outputs regenerate without training
- tests pass

## 5. Rebuild Dataset Files From Source

The retained dataset files are:

```text
tractatus_structure_latents/data/tractatus.json
tractatus_structure_latents/data/tractatus_bilingual.json
```

Rebuild them from the configured source pathway:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus.json \
  --languages en
```

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```

Then run:

```bash
python3 tools/verify_canonical_evidence.py
python3 -m pytest
```

Expected result: the rebuilt datasets preserve the expected proposition counts,
bilingual pairing checks, and downstream validation tests.

Note: if the dataset builder downloads source text, this stage requires network
access. For a fully offline test, provide a local source text with
`--source PATH`.

## 6. Smoke-Test Training Commands In A Scratch Output Root

Before a full rerun, test that each empirical runner can execute without
touching the canonical `results/dsh_validation/` tree:

```bash
python3 tools/phase1_ablations.py run \
  --out-root tmp/repro_smoke/phase1_ablations \
  --conditions full_model \
  --seeds 0 \
  --epochs 1
```

```bash
python3 tools/phase2_family_holdout.py run \
  --out-root tmp/repro_smoke/phase2_family_holdout \
  --conditions full_model \
  --folds 0 \
  --seeds 0 \
  --epochs 1
```

```bash
python3 tools/phase3_controlled_alignment.py run \
  --out-root tmp/repro_smoke/phase3_controlled_alignment \
  --batching paired \
  --conditions full_model \
  --lambdas 0.0 \
  --seeds 0 \
  --epochs 1
```

Expected result: each command completes and writes local ignored smoke outputs
under `tmp/repro_smoke/`.

Short smoke runs may emit sklearn warnings such as `y_pred contains classes not
in y_true` during parent/depth metric calculation. These warnings are acceptable
for 1-epoch smoke runs; treat only a non-zero exit code or missing output files
as a smoke-test failure.

## 7. Full Empirical Rerun In A Disposable Clone

Use a disposable clone or branch for the full empirical rerun. These commands
can take substantial time and will regenerate heavy ignored artifacts such as
checkpoints, raw parquet outputs, logs, and per-seed metrics.

Phase 1:

```bash
python3 tools/phase1_ablations.py run --skip-existing
```

Phase 2:

```bash
python3 tools/phase2_family_holdout.py run --skip-existing
```

Phase 3:

```bash
python3 tools/phase3_controlled_alignment.py run \
  --batching paired \
  --conditions full_model \
  --skip-existing
python3 tools/phase3_controlled_alignment.py successor-control --skip-existing
```

Phase 4:

```bash
python3 tools/phase4_case_studies.py run
```

Phase 4 intentionally has no `--skip-existing` option. By default it reuses the
existing `candidate_manifest_pre_text.csv` so the frozen text-blind case
selection is preserved. Use `--reselect` only when deliberately creating a new
frozen selection manifest.

Rebuild derived paper outputs:

```bash
python3 tools/build_canonical_reports.py
python3 tools/verify_canonical_evidence.py
python3 tools/export_paper_tables.py
python3 -m tractatus_structure_latents.evaluation.generate_paper_figures
python3 tools/build_final_paper_manifest.py
python3 tools/validate_final_paper_manifest.py
python3 tools/validate_paper_figure_manifest.py --require-validation
python3 -m pytest
```

The paper-figure generator writes the main manuscript figure under
`paper/figures/` and validation figures under `paper/figures/validation/`.
The validation rasters are expected to carry 600-dpi metadata, and the vector
PDF companions are tracked in `paper/final_paper_manifest.csv`.

Expected result: all canonical verification and tests pass. Some numerical
outputs may differ if dependency versions, hardware, random number behaviour, or
source text inputs differ from the archived run. Any difference should be
recorded before replacing retained evidence.

## 8. Final Review Before Updating Evidence

After a full rerun, review:

```bash
git status --short --ignored
git diff -- results/dsh_validation paper/tables paper/figures paper/final_paper_manifest.csv
```

Expected result:

- tracked summary/report/table/figure changes are explainable
- ignored checkpoint/raw/smoke outputs are not staged
- no heavy artifacts are added back to Git

Only commit regenerated small evidence after reviewing the diffs. Upload new
heavy artifacts as a new GitHub Release or external archive, then regenerate
`docs/heavy_artifacts_manifest.csv` if the archive contents change.
