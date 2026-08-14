# Tractatus Structure Latents

This repository is the companion repository for the empirical work behind the
final paper.

The project tests whether formal proposition-number relations in Wittgenstein's
*Tractatus Logico-Philosophicus* can supervise split-latent models toward a
structure-oriented representation. It does not claim to process logic or learn
general philosophical logic. The implemented supervision comes from parent,
depth, next-proposition, child-count, family, and bilingual same-id relations
derivable from proposition numbers.

## Reader Path

For the final paper, start here:

```text
paper/final_paper_manifest.csv
paper/figures/figure_manifest.csv
paper/tables/table_manifest.csv
results/dsh_validation/CANONICAL_VERIFICATION_REPORT.md
```

The canonical evidence layer is `results/dsh_validation/`, organised as Phase
1-4 retained outputs:

```text
results/dsh_validation/phase1_ablations/
results/dsh_validation/phase2_family_holdout/
results/dsh_validation/phase3_controlled_alignment/
results/dsh_validation/phase4_case_studies/
results/dsh_validation/canonical_reports/
```

Paper-facing figures include the main manuscript figure under `paper/figures/`
and validation figures under `paper/figures/validation/`. The validation
figures are regenerated from retained empirical outputs by the safe
reproduction wrapper; the raster exports are written at 600 dpi.

GitHub is kept as the clean code plus small-evidence repository. Heavy retained
artifacts such as checkpoints, raw/per-proposition parquet outputs, smoke
outputs, and generated validation-bundle copies are external release/archive
artifacts listed in [docs/heavy_artifacts_manifest.csv](docs/heavy_artifacts_manifest.csv)
and described in [docs/heavy_artifacts.md](docs/heavy_artifacts.md).

## Safe Reproduction

Use the final-paper wrapper for non-training reproduction and verification:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

Review the printed command list, then run the wrapper without `--dry-run` if
you want to regenerate the paper-facing derivatives from retained evidence. The
wrapper verifies or rebuilds canonical reports, paper tables, validation
figures, figure manifests, and final-paper manifests. It does not retrain models
or restore external checkpoint/raw artifacts.

For a staged clean-clone protocol that also covers external artifact
restoration, dataset rebuilding, scratch smoke runs, and full empirical reruns,
use [Full reproducibility test plan](docs/full_reproducibility_test_plan.md).

## Deliberate Retraining

Retraining is separate from the default paper reproduction path. The empirical
phase commands are documented in [docs/reproduction.md](docs/reproduction.md)
and should be run deliberately because they can write checkpoints, logs, raw
outputs, per-seed metrics, and figures under `results/dsh_validation/`.

## Documentation

- [Documentation index](docs/DOCUMENTATION.md)
- [Reproduction](docs/reproduction.md)
- [Full reproducibility test plan](docs/full_reproducibility_test_plan.md)
- [Results](docs/results.md)
- [Paper notes](paper/PAPER.md)
- [Repository layout for paper](docs/repository_layout_for_paper.md)
- [Dataset](docs/dataset.md)
- [Model](docs/model.md)
- [Experiments](docs/experiments.md)
- [Run artifacts](runs/RUN_ARTIFACTS.md)

## Repository Layout

```text
paper/                   manuscript, final manifest, final figures, table exports, and paper notes
results/dsh_validation/  Phase 1-4 canonical evidence, reports, and bundles
tractatus_structure_latents/
                         importable package for data, model, training, and evaluation code
tools/                   empirical phase CLIs, report builders, validators, and wrappers
tests/                   narrow validation and regression tests
docs/                    reader and reproduction documentation
```

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The retained datasets used by the current evidence layer are:

```text
tractatus_structure_latents/data/tractatus.json
tractatus_structure_latents/data/tractatus_bilingual.json
```

Dataset rebuild commands are documented in [docs/reproduction.md](docs/reproduction.md).

## License

This repository uses a mixed-license structure:

- Code is licensed under [Apache-2.0](LICENSE).
- Documentation and paper text are licensed under [CC BY 4.0](LICENSE-DOCS.md).
- Tractatus-derived text/data are covered by the source-text notice in [DATA_NOTICE.md](DATA_NOTICE.md).

The repository does not claim copyright over the underlying public-domain
Tractatus text.
