# Free-Energy-Style Latent Transition Prototype

This folder contains exploratory scaffolding for testing whether exported
Tractatus structure latents can support a simple sequential dynamics model. The
main script, `free_energy_latent_transition.py`, performs a gradient-based belief
update over one latent state so that a trained transition model better predicts
the observed next latent state.

This is not part of the canonical monolingual or bilingual seed-sweep workflow.
The completed canonical workflow remains split-latent VAE training, structure
evaluation, seed-sweep aggregation, and figure generation under
`runs/seed_sweeps/`.

## Components

```text
tractatus_structure_latents/models/dynamics.py
  Defines GRUTransition, a GRU transition model over latent vectors.

tractatus_structure_latents/training/train_dynamics.py
  Trains GRUTransition on an exported latent sequence z[0:T-1] -> z[1:T].

tractatus_structure_latents/active_inference/free_energy_latent_transition.py
  Optimizes a single latent belief state against a free-energy-style objective.
```

`GRUTransition` maps a latent sequence to a predicted next-latent distribution:

```text
input:  z_sequence shaped [batch, time, latent_dim]
output: mu, logvar shaped [batch, time, latent_dim]
```

The current prototype uses the predicted mean `mu`. The predicted `logvar` is
computed by the model but is not yet used in the free-energy objective.

## Objective

For a selected proposition index `i`, the script starts from the exported latent
state `z[i]` and treats it as an optimizable belief vector. The transition model
predicts the next latent state from this belief:

```text
predicted_next = transition.next_distribution(belief)
```

The observed next latent is `z[i + 1]`. The optimized scalar objective is:

```text
free_energy = prediction_error + complexity
prediction_error = squared_error(predicted_next, z[i + 1])
complexity = 0.5 * precision * ||belief||^2
```

In code, this is implemented as summed MSE plus an L2 penalty on the belief
vector. The update is therefore best understood as a free-energy-style latent
state adjustment, not as a full active-inference agent.

## Expected Inputs

```text
--latents   exported latent tensor from evaluate_structure.py
--dynamics  separately trained GRUTransition state dict
--index     proposition index whose next latent is observed
```

The latent tensor should be shaped:

```text
[T, latent_dim]
```

For the paper's split-latent experiments, the intended input is usually an
exported `z_structure` tensor rather than the full text-and-structure latent.

## Example Workflow

Train a transition model from monolingual structure latents:

```bash
python3 -m tractatus_structure_latents.training.train_dynamics \
  --latents runs/seed_sweeps/monolingual_split_24_8_reg005/latents/seed000_structure.pt \
  --out runs/dynamics/monolingual_seed000_structure.pt \
  --epochs 200 \
  --lr 0.001
```

Run the free-energy-style belief update at proposition index `100`:

```bash
python3 -m tractatus_structure_latents.active_inference.free_energy_latent_transition \
  --latents runs/seed_sweeps/monolingual_split_24_8_reg005/latents/seed000_structure.pt \
  --dynamics runs/dynamics/monolingual_seed000_structure.pt \
  --index 100
```

The script prints:

```text
initial_free_energy  free energy before belief optimization
final_free_energy    free energy after belief optimization
belief_norm          L2 norm of the optimized latent belief
```

## Bilingual Ordering Caveat

Do not run this script directly on bilingual latent exports unless the tensor is
first filtered into a single-language proposition sequence. Bilingual evaluation
exports are interleaved by proposition id, for example:

```text
1/en, 1/de, 1.1/en, 1.1/de, ...
```

Adjacent rows in that tensor are often translation pairs, not successive
propositions. Directly training transition dynamics on the raw bilingual tensor
would therefore mix translation alignment with next-proposition dynamics.

## Current Status

No canonical dynamics checkpoints are currently included. This folder should be
read as experimental scaffolding rather than a reported result.

The prototype currently lacks:

```text
held-out transition evaluation
baseline transition models
multi-seed dynamics sweeps
use of predicted uncertainty/log variance
language-filtered bilingual transition experiments
paper-ready aggregate metrics
```

For the current paper, this belongs in future work rather than the main results.
A defensible paper extension would need held-out transition metrics, baseline
comparisons, and clear separation between monolingual next-proposition dynamics
and bilingual same-id alignment effects.
