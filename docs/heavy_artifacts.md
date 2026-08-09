# Heavy External Artifacts

GitHub is used for code, documentation, manifests, final manuscript-facing
figures/tables, compact canonical reports, and small retained source data.
Large empirical artifacts are kept outside Git and identified by SHA256 hashes.

## Current External Archive

The current local archive prepared for release upload is:

```text
/tmp/tractatus_heavy_artifacts_2026-08-09/tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
```

Archive SHA256:

```text
731891afbe1ebc300c5f3610838b943994745c51167b28917e0562a9fb4bf576
```

The per-file manifest tracked in Git is:

```text
docs/heavy_artifacts_manifest.csv
```

It lists each removed artifact path, artifact class, bytes, file SHA256, and the
archive file expected to contain it.

## Artifact Classes Kept Outside Git

- Model checkpoints under `results/dsh_validation/**/checkpoints/`
- Raw and per-proposition parquet outputs under `results/dsh_validation/`
- Smoke outputs under `results/dsh_validation/**/smoke/`
- Generated validation bundle copies:
  `results/dsh_validation/dsh_validation_bundle/` and
  `results/dsh_validation/dsh_validation_bundle.zip`

These files are retained evidence or reproducibility byproducts, but they are
too large or duplicative for the main Git history.

## Restore Instructions

Download the archive from the project release/archive record, then verify it:

```bash
sha256sum tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
```

The hash must match the archive SHA256 above. Restore into a clean checkout:

```bash
tar -xzf tractatus-heavy-retained-artifacts-2026-08-09.tar.gz
```

Then verify the restored files against the tracked manifest:

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

## Upload Instructions

Attach `tractatus-heavy-retained-artifacts-2026-08-09.tar.gz` to a GitHub
Release, Zenodo record, OSF project, or institutional archive. Publish the
archive hash and keep the filename unchanged unless
`docs/heavy_artifacts_manifest.csv` is regenerated.
