# ACoT-VLA JAX to PyTorch Review Map

This note is meant to make the PR easier to review. It maps the ACoT-VLA JAX implementation in the sibling `../ACoT-VLA` checkout to the PyTorch implementation added in this branch.

## Short Version

The upstream ACoT-VLA implementation adds ACoT behavior to the existing JAX train path instead of creating a separate training script. This branch follows the same shape for PyTorch:

- Keep training in `scripts/train_pytorch.py`.
- Add an `ACOTConfig` model config.
- Add `ACOTPytorch` as the PyTorch model implementation.
- Reuse the existing PI0 PyTorch preprocessing, prefix embedding, time/noise sampling, checkpoint helper, and train loop infrastructure.
- Extend the existing data path to emit `coarse_actions` for ACoT batches.

## Main Source Map

| JAX reference | PyTorch implementation | What to check |
| --- | --- | --- |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::ACOTConfig` | `src/openpi/models/acot_config.py::ACOTConfig` | ACoT-specific horizons, dual expert variants, reasoner flags, model type, fake/coarse action specs. |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::ACOT_VLA.__init__` | `src/openpi/models_pytorch/acot_vla_pytorch.py::ACOTPytorch.__init__` | PaliGemma plus two Gemma experts, coarse/final action projections, pi0/pi05 time conditioning, explicit and implicit reasoner modules. |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::embed_prefix` | inherited from `src/openpi/models_pytorch/pi0_pytorch.py::PI0Pytorch.embed_prefix` | Image/language prefix construction is intentionally reused from existing PI0 PyTorch. |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::embed_suffix` | `src/openpi/models_pytorch/acot_vla_pytorch.py::ACOTPytorch.embed_suffix` | Coarse reasoner stream vs final expert stream, time embeddings, pi05 adaRMS condition, pi0 state token path. |
| JAX explicit/implicit reasoner blocks inside `embed_suffix` | `src/openpi/models_pytorch/acot_vla_pytorch.py::_apply_reasoning` | Explicit coarse-action cross attention, implicit KV-token cross attention, and fusion behavior. |
| JAX `LearnableQueryExtractor`, `AttentionPoolingExtractor`, `DownsampleExtractor` | `src/openpi/models_pytorch/acot_vla_pytorch.py::{LearnableQueryExtractor,AttentionPoolingExtractor,DownsampleExtractor}` | KV-cache-to-token implicit reasoner options. The PyTorch version groups layers similarly and uses torch modules. |
| JAX three-stream PaliGemma call `[prefix, coarse, expert]` | `src/openpi/models_pytorch/gemma_pytorch.py::PaliGemmaWithExpertModel` | The shared PyTorch Gemma wrapper now supports one or multiple expert streams. PI0 still uses one expert; ACoT uses two. |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::compute_loss` | `src/openpi/models_pytorch/acot_vla_pytorch.py::ACOTPytorch.forward` | Flow matching loss for final actions, plus optional explicit coarse-action reasoner loss. |
| `../ACoT-VLA/src/openpi/models/acot_vla.py::sample_actions` | `src/openpi/models_pytorch/acot_vla_pytorch.py::ACOTPytorch.sample_actions`, `_denoise`, `denoise_step` | Coarse denoising first when explicit reasoning is enabled, then final action denoising with explicit/implicit reasoning. |
| `../ACoT-VLA/scripts/train.py::acot_train_step` | `scripts/train_pytorch.py` ACoT branch in model creation and train loop | PyTorch keeps one train script and calls `model(observation, actions, coarse_actions)` when the batch has three items. |

## Model Architecture Mapping

### Three-stream language/action model

JAX uses one PaliGemma prefix stream plus two action-expert streams:

- stream 0: image/language prefix
- stream 1: coarse action reasoner
- stream 2: final action expert

The PyTorch side implements this through `PaliGemmaWithExpertModel`:

- `src/openpi/models_pytorch/gemma_pytorch.py` accepts either one expert config or a sequence of expert configs.
- `PI0Pytorch` remains compatible through `self.gemma_expert`.
- `ACOTPytorch` passes `[coarse_action_expert_config, action_expert_config]`, so it gets two expert streams.

This avoids keeping a separate duplicated `PaliGemmaWithDualExpertModel` inside the ACoT file.

### ACoT modules

The following JAX `nnx` modules are represented as torch modules:

| JAX idea | PyTorch location |
| --- | --- |
| `MLP` | `src/openpi/models_pytorch/acot_vla_pytorch.py::MLP` |
| `UnifiedAttentionModule` | `src/openpi/models_pytorch/acot_vla_pytorch.py::UnifiedAttentionModule` |
| `LearnableQueryExtractor` | `src/openpi/models_pytorch/acot_vla_pytorch.py::LearnableQueryExtractor` |
| `AttentionPoolingExtractor` | `src/openpi/models_pytorch/acot_vla_pytorch.py::AttentionPoolingExtractor` |
| `DownsampleExtractor` | `src/openpi/models_pytorch/acot_vla_pytorch.py::DownsampleExtractor` |
| coarse action input/output projections | `ACOTPytorch.coarse_action_in_proj`, `coarse_action_out_proj` |
| final action input/output projections | `ACOTPytorch.action_in_proj`, `action_out_proj` |
| pi05 adaRMS time MLPs | `coarse_time_mlp_*`, `time_mlp_*` |
| pi0 action/time MLPs | `coarse_action_time_mlp_*`, `action_time_mlp_*` |

## Training Path Mapping

The JAX implementation adds a separate `acot_train_step` but keeps it inside `scripts/train.py`. This branch mirrors that by extending existing PyTorch training code:

- `scripts/train_pytorch.py` instantiates `ACOTPytorch` when `config.model` is `ACOTConfig`.
- The train loop accepts either `(observation, actions)` or `(observation, actions, coarse_actions)`.
- For ACoT, it sends the third item to `ACOTPytorch.forward`.
- The loss returned by `ACOTPytorch.forward` is already a scalar torch tensor.

There is no new PyTorch training entrypoint.

## Data Path Mapping

The JAX ACoT path creates both final `actions` and coarse `coarse_actions`. This branch ports that behavior into the shared transform/data loader path:

| Purpose | PyTorch branch files |
| --- | --- |
| Create coarse/final action chunks with separate horizons/strides | `src/openpi/transforms.py::GenerateACOTActions` |
| Apply delta/absolute transforms to both action keys | `src/openpi/transforms.py::ACOTDeltaActions`, `ACOTAbsoluteActions` |
| Pad both action keys to model action dim | `src/openpi/transforms.py::ACOTPadStatesAndActions` |
| Reuse action norm stats for coarse actions when needed | `src/openpi/transforms.py::_with_acot_coarse_action_stats` |
| Increase LeRobot action chunk size for both horizons | `src/openpi/training/data_loader.py::create_torch_dataset` |
| Yield a three-item batch for ACoT | `src/openpi/training/data_loader.py::TorchDataLoader.__iter__` |
| Smoke-test fake ACoT batches and specs | `src/openpi/training/data_loader_test.py` |

## Inference and Policy Output Mapping

JAX `sample_actions` can return both `actions` and `coarse_actions` when explicit reasoning is enabled. The PyTorch branch preserves that shape:

- `ACOTPytorch.sample_actions` returns `{"actions": actions, "coarse_actions": explicit_action_reason}` when explicit reasoning is enabled.
- Policy output transforms were updated to pass through `coarse_actions` where applicable:
  - `src/openpi/policies/aloha_policy.py`
  - `src/openpi/policies/droid_policy.py`
  - `src/openpi/policies/libero_policy.py`
  - `src/openpi/policies/policy.py`
  - `src/openpi/policies/policy_config.py`

## Review Notes

The most important review path is:

1. `src/openpi/models/acot_config.py`
2. `src/openpi/models_pytorch/acot_vla_pytorch.py`
3. `src/openpi/models_pytorch/gemma_pytorch.py`
4. `scripts/train_pytorch.py`
5. `src/openpi/transforms.py`
6. `src/openpi/training/data_loader.py`

The second commit intentionally refactors the first ACoT PyTorch implementation so the ACoT model is thinner:

- Common PI0 PyTorch helpers are inherited from `PI0Pytorch`.
- The multi-expert Gemma wrapper lives in `gemma_pytorch.py` instead of a one-off dual-expert class.
- ACoT-specific code remains focused on coarse/final action suffixes, reasoning fusion, loss, and denoising.

## Not Ported Here

These JAX-side pieces are not implemented as part of this PyTorch branch:

- JAX checkpoint weight conversion for ACoT-specific weights.
- The full upstream ACoT config zoo for every robot/domain.
- A JAX-compatible `ACOTConfig.create`; this branch supports the PyTorch model path.
- Exact numerical parity tests against the JAX implementation.
