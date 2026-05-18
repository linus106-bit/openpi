import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils


@dataclasses.dataclass(frozen=True)
class ACOTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    coarse_action_expert_variant: _gemma.Variant = "gemma_300m"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    action_dim: int = 32
    coarse_action_horizon: int = 50
    action_horizon: int = 30
    max_token_len: int = None  # type: ignore

    pi05: bool = True
    discrete_state_input: bool = None  # type: ignore
    pytorch_compile_mode: str | None = "max-autotune"

    adopt_explicit_action_reasoner: bool = False
    adopt_implicit_action_reasoner: bool = False
    query_based_implicit_extractor: bool = False
    attention_pooling_implicit_extractor: bool = False
    downsample_based_implicit_extractor: bool = False

    # Used by the generic ACOT data transform to split a long action chunk into
    # coarse and final action targets.
    coarse_action_stride: int = 2
    action_stride: int = 1

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        selected_extractors = sum(
            [
                self.query_based_implicit_extractor,
                self.attention_pooling_implicit_extractor,
                self.downsample_based_implicit_extractor,
            ]
        )
        if self.adopt_implicit_action_reasoner and selected_extractors != 1:
            raise ValueError("Exactly one implicit extractor must be selected when implicit reasoning is enabled.")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.ACOT_VLA_PI05 if self.pi05 else _model.ModelType.ACOT_VLA_PI0

    @override
    def create(self, rng: at.KeyArrayLike):
        raise NotImplementedError("ACOTConfig currently supports the PyTorch model path only.")

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def coarse_actions_spec(self, *, batch_size: int = 1) -> _model.CoarseActions:
        return jax.ShapeDtypeStruct([batch_size, self.coarse_action_horizon, self.action_dim], jnp.float32)

    def fake_coarse_act(self, batch_size: int = 1) -> _model.CoarseActions:
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), self.coarse_actions_spec(batch_size=batch_size))

    def get_freeze_filter(
        self,
        freeze_llm: bool = False,
        freeze_llm_embedder: bool = True,
        freeze_vision: bool = False,
        freeze_dual_ae: tuple[bool, bool] = (False, False),
    ) -> nnx.filterlib.Filter:
        freeze_paths = []
        if freeze_vision:
            freeze_paths.append(nnx_utils.PathRegex(".*vision_tower.*|.*img.*"))
        if freeze_llm:
            freeze_paths.append(nnx_utils.PathRegex(".*paligemma.*language_model.*"))
        if freeze_dual_ae[0]:
            freeze_paths.append(nnx_utils.PathRegex(".*gemma_coarse_expert.*"))
        if freeze_dual_ae[1]:
            freeze_paths.append(nnx_utils.PathRegex(".*gemma_expert.*"))
        if not freeze_paths:
            return nnx.Nothing

        base_freeze_filter = nnx.Any(*freeze_paths)
        keep_alive_paths = []
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            keep_alive_paths.append(nnx_utils.PathRegex(".*lora.*"))
        if freeze_llm and not freeze_llm_embedder:
            keep_alive_paths.append(nnx_utils.PathRegex(".*embed.*|.*embedding.*"))
        if not keep_alive_paths:
            return base_freeze_filter
        return nnx.All(base_freeze_filter, nnx.Not(nnx.Any(*keep_alive_paths)))
