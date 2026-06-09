# runs

This folder contains generated experiment artifacts and run documentation.

The canonical generated artifacts should live under:

```text
runs/seed_sweeps/
```

Avoid adding loose checkpoints, metrics, latent tensors, or plots directly to the root of `runs/`. Older root-level retained artifacts have been superseded by repeated-seed sweeps.

## Canonical Contents

```text
runs/
  RUN_ARTIFACTS.md
  seed_sweeps/
    SEED_SWEEPS.md
    monolingual_split_24_8_reg005/
    bilingual_alignment_lambda_sweep/
```

## Regeneration

Use the root [README.md](../README.md) for the full regeneration workflow:

```text
1. Build datasets.
2. Run monolingual seeds 0..9.
3. Run bilingual lambda values 0.00, 0.03, 0.10, 0.30, 1.00 across seeds 0..9.
4. Aggregate metrics.
5. Generate paper figures.
```

## Artifact Policy

Keep:

```text
runs/RUN_ARTIFACTS.md
runs/seed_sweeps/SEED_SWEEPS.md
runs/seed_sweeps/monolingual_split_24_8_reg005/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
```

Do not keep old one-off root artifacts unless they are intentionally archived and documented.
