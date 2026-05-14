# ACoT-VLA PyTorch Migration Plan

This repo integrates ACoT-VLA through the PyTorch training and inference path.

## Implemented Scope

- Added `ModelType.ACOT_VLA_PI0` and `ModelType.ACOT_VLA_PI05`.
- Added `ACOTConfig` with coarse/final horizons, dual expert variants, and explicit/implicit reasoner flags.
- Added `ACOTPytorch`, which extends the PI0 PyTorch pattern to three streams:
  - PaliGemma prefix model
  - coarse action expert
  - final action expert
- Added torch ports of the ACoT helper modules:
  - `MLP`
  - `UnifiedAttentionModule`
  - implicit cache extractors
- Extended the PyTorch trainer to pass `(observation, actions, coarse_actions)` batches to ACOT models.
- Extended the data path to create and return `coarse_actions`.
- Extended policy inference to preserve ACOT dict outputs, including `coarse_actions` when explicit reasoning is enabled.

## Out of Scope

- JAX ACOT checkpoint conversion to PyTorch.
- Exact numerical parity against the ACoT-VLA JAX implementation.
- Performance tuning beyond smoke-testable parity.

## Example Commands

```bash
uv run scripts/train_pytorch.py acot_icra_simulation_challenge_reasoning_to_action_torch --exp_name <run>
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py acot_icra_simulation_challenge_reasoning_to_action_torch --exp_name <run>
```

For local smoke testing:

```bash
uv run scripts/train_pytorch.py debug_acot --exp_name smoke
```
