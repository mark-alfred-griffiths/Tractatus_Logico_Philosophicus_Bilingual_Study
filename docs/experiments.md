# Experiments

The current empirical layer for `paper/Tractatus_final.docx` is the Phase 1-4
validation suite under `results/dsh_validation/`.

## Canonical Phases

```text
results/dsh_validation/phase1_ablations/
results/dsh_validation/phase2_family_holdout/
results/dsh_validation/phase3_controlled_alignment/
results/dsh_validation/phase4_case_studies/
```

Phase roles:

- Phase 1: retained-corpus ablation diagnostics.
- Phase 2: immediate-parent-family holdout.
- Phase 3: controlled paired-batch bilingual alignment.
- Phase 4: frozen text-blind case selection.

## Safe Paper Reproduction

The default paper reproduction path does not retrain:

```bash
python3 tools/reproduce_final_paper_outputs.py --dry-run
```

After reviewing the printed command list, run without `--dry-run` to rebuild
paper-facing derivatives from retained Phase 1-4 outputs.

## Deliberate Reruns

Phase commands are deliberate reruns and can write checkpoints, logs, raw
outputs, per-seed metrics, reports, and figures:

```bash
python3 tools/phase1_ablations.py run --skip-existing
python3 tools/phase2_family_holdout.py run --skip-existing
python3 tools/phase3_controlled_alignment.py run --batching paired --conditions full_model --skip-existing
python3 tools/phase3_controlled_alignment.py successor-control --skip-existing
python3 tools/phase4_case_studies.py run
```

Rebuild summaries and checks after deliberate reruns:

```bash
python3 tools/build_canonical_reports.py
python3 tools/verify_canonical_evidence.py
python3 tools/export_paper_tables.py
```
