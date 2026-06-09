# Model

The main model is a split-latent recurrent VAE. It separates text reconstruction from hierarchy-structure learning by assigning each objective its own latent subspace.

Implementation lives in:

```text
tractatus_structure_latents/models/vae.py
```

## Split-Latent Architecture

The canonical model is `SplitLatentHierarchicalRNNVAE`.

```text
text tokens
  -> embedding
  -> bidirectional GRU encoder
  -> mu, logvar in R^32
  -> z = [z_text, z_structure]
```

Canonical dimensions:

```text
z_text:       24
z_structure:  8
total:        32
```

## z_text

`z_text` is the reconstruction subspace.

```text
z_text + language_id -> GRU decoder -> reconstructed text
```

For bilingual runs, the decoder receives a learned language embedding so it knows whether to reconstruct German or English.

## z_structure

`z_structure` is the hierarchy subspace. It is used by auxiliary structure heads:

```text
parent_head(z_structure)
depth_head(z_structure)
next_head(z_structure)
child_count_head(z_structure)
```

The decoder does not receive `z_structure`, which prevents the hierarchy subspace from becoming an unrestricted reconstruction channel.

## Bilingual Alignment

Bilingual rows are flattened into proposition-language samples:

```text
(id=4.121, language=en, text=...)
(id=4.121, language=de, text=...)
```

Both samples share the same proposition index and structural targets. The optional alignment loss pulls same-id German and English structure latents together:

```text
L_align = mse(z_structure_en(id), z_structure_de(id))
```

This loss is controlled by:

```bash
--lambda-language-alignment
```

The canonical bilingual lambda sweep evaluates:

```text
0.00, 0.03, 0.10, 0.30, 1.00
```

## Loss Function

The bilingual split model optimises:

```text
L = reconstruction_loss
  + beta_text * KL(z_text)
  + beta_structure * KL(z_structure)
  + lambda_parent * parent_loss
  + lambda_depth * depth_loss
  + lambda_next * next_loss
  + lambda_child * child_count_loss
  + lambda_language_alignment * alignment_loss
```

Canonical shared settings:

```text
beta_text = 0.01
beta_structure = 0.05
lambda_parent = 0.2
lambda_depth = 0.1
lambda_next = 0.2
lambda_child = 0.02
```

## Canonical Artifacts

```text
runs/seed_sweeps/monolingual_split_24_8_reg005/
runs/seed_sweeps/bilingual_alignment_lambda_sweep/
```

Each study stores checkpoints, logs, metrics, exported latents, summaries, and a `manifest.json` under its own folder.
