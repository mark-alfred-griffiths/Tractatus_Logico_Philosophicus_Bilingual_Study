# Paper Companion Files

This directory contains manuscript-facing files for the empirical work behind
the final paper. Manuscript text files should be treated as retained provenance
unless a current manifest marks them as active.

## Figures And Tables

Figure manifest:

```text
paper/figures/figure_manifest.csv
```

Table manifest:

```text
paper/tables/table_manifest.csv
```

The final-paper manifest tying manuscript-facing files to source artifacts is:

```text
paper/final_paper_manifest.csv
```

Figures and table CSVs are reproducible derivatives from retained evidence, not
the primary evidence layer. The current primary evidence lives under:

```text
results/dsh_validation/
```

## Reproduction

Use the safe non-training wrapper from the repository root:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

After reviewing the command list, run without `--dry-run` to regenerate or
verify final-paper derivatives from retained evidence.

For full clean-clone verification, external artifact restoration, dataset
rebuilds, and empirical reruns, follow
`docs/full_reproducibility_test_plan.md`.

Useful narrow checks:

```bash
python3 tools/validate_final_paper_manifest.py
python3 tools/validate_paper_figure_manifest.py
python3 -m unittest tests/test_paper_tables.py
```
