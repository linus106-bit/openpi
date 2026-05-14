import logging
import math
from typing import Literal

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import sample_beta


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, activate: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activate = activate

    def forward(self, x: Tensor) -> Tensor:
        if self.activate:
            return self.fc3(F.silu(self.fc2(F.silu(self.fc1(x)))))
        return self.fc3(self.fc2(self.fc1(x)))


class UnifiedAttentionModule(nn.Module):
    def __init__(
        self,
        in_dim_1: int,
        in_dim_2: int,
        out_dim: int,
        *,
        apply_sigmoid: bool,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.q_proj = nn.Linear(in_dim_1, hidden_dim)
        self.k_proj = nn.Linear(in_dim_2, hidden_dim)
        self.v_proj = nn.Linear(in_dim_2, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, feat_1: Tensor, feat_2: Tensor) -> Tensor:
        q = self.q_proj(feat_1)
        k = self.k_proj(feat_2)
        v = self.v_proj(feat_2)
        output, _ = self.attn(q, k, v, need_weights=False)
        output = self.fc_out(output)
        if self.apply_sigmoid:
            return torch.sigmoid(output)
        return output


class _GroupedAttentionExtractor(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        *,
        heads: int = 8,
        head_dim: int = 256,
        group_size: int = 3,
        downsample_dim: int | None = None,
        num_queries: int = 1,
        query_mode: Literal["learned", "mean"] = "learned",
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = depth
        self.heads = heads
        self.group_size = group_size
        self.num_groups = math.ceil(depth / group_size)
        self.downsample_dim = downsample_dim or heads * head_dim
        if self.downsample_dim % heads != 0:
            raise ValueError("downsample_dim must be divisible by heads")
        self.head_dim = self.downsample_dim // heads
        self.num_queries = num_queries
        self.query_mode = query_mode

        if query_mode == "learned":
            self.query_params = nn.Parameter(torch.randn(depth, num_queries, dim) * 0.02)
        else:
            self.query_params = None

        self.q_proj = nn.ModuleList(nn.Linear(dim, self.downsample_dim) for _ in range(self.num_groups))
        self.k_proj = nn.ModuleList(nn.Linear(dim, self.downsample_dim) for _ in range(self.num_groups))
        self.v_proj = nn.ModuleList(nn.Linear(dim, self.downsample_dim) for _ in range(self.num_groups))
        self.out_proj = nn.ModuleList(nn.Linear(self.downsample_dim, output_dim) for _ in range(self.num_groups))

    def forward(self, keys: Tensor, values: Tensor) -> Tensor:
        batch_size, depth, tokens, _ = keys.shape
        if depth != self.depth:
            raise ValueError(f"Expected {self.depth} cache layers, got {depth}.")

        outputs = []
        for layer_idx in range(depth):
            group_idx = layer_idx // self.group_size
            key = keys[:, layer_idx]
            value = values[:, layer_idx]
            if self.query_mode == "mean":
                query = key.mean(dim=1, keepdim=True)
            else:
                query = self.query_params[layer_idx][None, :, :].expand(batch_size, -1, -1)

            q = self.q_proj[group_idx](query).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
            k = self.k_proj[group_idx](key).view(batch_size, tokens, self.heads, self.head_dim).transpose(1, 2)
            v = self.v_proj[group_idx](value).view(batch_size, tokens, self.heads, self.head_dim).transpose(1, 2)

            attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim), dim=-1)
            pooled = torch.matmul(attn, v)
            pooled = pooled.mean(dim=2)
            pooled = pooled.transpose(1, 2).reshape(batch_size, self.downsample_dim)
            outputs.append(self.out_proj[group_idx](pooled))
        return torch.stack(outputs, dim=1)


class LearnableQueryExtractor(_GroupedAttentionExtractor):
    def __init__(self, num_queries: int, dim: int, output_dim: int, depth: int, *, heads: int, head_dim: int):
        super().__init__(
            dim,
            output_dim,
            depth,
            heads=heads,
            head_dim=head_dim,
            num_queries=num_queries,
            query_mode="learned",
        )


class AttentionPoolingExtractor(_GroupedAttentionExtractor):
    def __init__(self, dim: int, output_dim: int, depth: int, *, heads: int, head_dim: int):
        super().__init__(dim, output_dim, depth, heads=heads, head_dim=head_dim, query_mode="mean")


class DownsampleExtractor(_GroupedAttentionExtractor):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        depth: int,
        *,
        downsample_dim: int,
        heads: int,
        num_queries: int = 1,
    ):
        super().__init__(
            dim,
            output_dim,
            depth,
            heads=heads,
            head_dim=max(1, downsample_dim // heads),
            downsample_dim=downsample_dim,
            num_queries=num_queries,
            query_mode="learned",
        )


class PaliGemmaWithDualExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        coarse_action_expert_config,
        action_expert_config,
        *,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()
        if use_adarms is None:
            use_adarms = [False, False, False]

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_coarse_expert = self._make_expert(coarse_action_expert_config, use_adarms[1])
        self.gemma_expert = self._make_expert(action_expert_config, use_adarms[2])
        self.gemma_coarse_expert.model.embed_tokens = None
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)

    def _make_expert(self, config, use_adarms: bool):
        config_hf = CONFIG_MAPPING["gemma"](
            head_dim=config.head_dim,
            hidden_size=config.width,
            intermediate_size=config.mlp_dim,
            num_attention_heads=config.num_heads,
            num_hidden_layers=config.depth,
            num_key_value_heads=config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms,
            adarms_cond_dim=config.width if use_adarms else None,
        )
        return GemmaForCausalLM(config=config_hf)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.to(dtype=torch.bfloat16)
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: Tensor) -> Tensor:
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[Tensor | None] | None = None,
    ):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        if adarms_cond is None:
            adarms_cond = [None] * len(inputs_embeds)

        active_indices = [idx for idx, embeds in enumerate(inputs_embeds) if embeds is not None]
        if active_indices == [0]:
            output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            return [output.last_hidden_state, None, None], output.past_key_values

        if active_indices in ([1], [2]):
            expert = self.gemma_coarse_expert.model if active_indices == [1] else self.gemma_expert.model
            output = expert.forward(
                inputs_embeds=inputs_embeds[active_indices[0]],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[active_indices[0]],
            )
            outputs = [None, None, None]
            outputs[active_indices[0]] = output.last_hidden_state
            return outputs, output.past_key_values

        return self._forward_joint(attention_mask, position_ids, inputs_embeds, adarms_cond), None

    def _forward_joint(
        self,
        attention_mask: Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: list[Tensor | None],
        adarms_cond: list[Tensor | None],
    ):
        models = [
            self.paligemma.language_model,
            self.gemma_coarse_expert.model,
            self.gemma_expert.model,
        ]
        active_indices = [idx for idx, embeds in enumerate(inputs_embeds) if embeds is not None]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        hidden_states_by_stream = list(inputs_embeds)

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            gates = {}
            for stream_idx in active_indices:
                layer = models[stream_idx].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(
                    hidden_states_by_stream[stream_idx], cond=adarms_cond[stream_idx]
                )
                gates[stream_idx] = gate
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )

            scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                self.paligemma.language_model.layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling,
            )
            batch_size = query_states.shape[0]
            att_output = att_output.reshape(batch_size, -1, query_states.shape[1] * query_states.shape[-1])

            next_hidden_states = list(hidden_states_by_stream)
            start_pos = 0
            for stream_idx in active_indices:
                hidden_states = hidden_states_by_stream[stream_idx]
                layer = models[stream_idx].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]
                stream_att = att_output[:, start_pos:end_pos]
                if stream_att.dtype != layer.self_attn.o_proj.weight.dtype:
                    stream_att = stream_att.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(stream_att)
                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[stream_idx])  # noqa: SLF001
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[stream_idx])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                next_hidden_states[stream_idx] = modeling_gemma._gated_residual(  # noqa: SLF001
                    after_first_residual, out_emb, gate
                )
                start_pos = end_pos
            hidden_states_by_stream = next_hidden_states

        outputs = [None, None, None]
        for stream_idx in active_indices:
            out_emb, _ = models[stream_idx].norm(hidden_states_by_stream[stream_idx], cond=adarms_cond[stream_idx])
            outputs[stream_idx] = out_emb
        return outputs


class ACOTPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.coarse_action_horizon = config.coarse_action_horizon
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        coarse_action_expert_config = _gemma.get_config(config.coarse_action_expert_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithDualExpertModel(
            paligemma_config,
            coarse_action_expert_config,
            action_expert_config,
            use_adarms=[False, True, True] if self.pi05 else [False, False, False],
            precision=config.dtype,
        )

        self.coarse_action_in_proj = nn.Linear(config.action_dim, coarse_action_expert_config.width)
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.coarse_action_out_proj = nn.Linear(coarse_action_expert_config.width, config.action_dim)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.coarse_time_mlp_in = nn.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.coarse_time_mlp_out = nn.Linear(coarse_action_expert_config.width, coarse_action_expert_config.width)
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.coarse_action_time_mlp_in = nn.Linear(
                2 * coarse_action_expert_config.width, coarse_action_expert_config.width
            )
            self.coarse_action_time_mlp_out = nn.Linear(
                coarse_action_expert_config.width, coarse_action_expert_config.width
            )
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.adopt_explicit_action_reasoner = config.adopt_explicit_action_reasoner
        if self.adopt_explicit_action_reasoner:
            self.explicit_action_reasoner = UnifiedAttentionModule(
                action_expert_config.width,
                coarse_action_expert_config.width,
                action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
            )

        self.adopt_implicit_action_reasoner = config.adopt_implicit_action_reasoner
        if self.adopt_implicit_action_reasoner:
            if config.query_based_implicit_extractor:
                self.implicit_action_reasoner = LearnableQueryExtractor(
                    num_queries=8,
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                )
            elif config.attention_pooling_implicit_extractor:
                self.implicit_action_reasoner = AttentionPoolingExtractor(
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    heads=paligemma_config.num_heads,
                    head_dim=paligemma_config.head_dim,
                )
            elif config.downsample_based_implicit_extractor:
                self.implicit_action_reasoner = DownsampleExtractor(
                    dim=paligemma_config.head_dim,
                    output_dim=action_expert_config.width,
                    depth=paligemma_config.depth,
                    downsample_dim=max(paligemma_config.num_heads, paligemma_config.head_dim // 2),
                    heads=paligemma_config.num_heads,
                )
            else:
                raise ValueError("At least one implicit extractor type must be selected.")
            self.implicit_action_reasoner_interact = UnifiedAttentionModule(
                action_expert_config.width,
                action_expert_config.width,
                action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
            )

        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            self.explicit_action_reason_proj = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.implicit_action_reason_proj = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_reasoning_fusion = UnifiedAttentionModule(
                2 * action_expert_config.width,
                2 * action_expert_config.width,
                action_expert_config.width,
                hidden_dim=action_expert_config.width,
                apply_sigmoid=False,
            )
        elif self.adopt_explicit_action_reasoner or self.adopt_implicit_action_reasoner:
            self.action_reasoning_fusion = MLP(
                2 * action_expert_config.width, action_expert_config.width, action_expert_config.width, activate=False
            )

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False
        msg = (
            "transformers_replace is not installed correctly. Please install it with "
            "`uv pip install transformers==4.53.2` and `cp -r "
            "./src/openpi/models_pytorch/transformers_replace/* "
            ".venv/lib/python3.11/site-packages/transformers/`."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_coarse_expert.model.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for ACOTPytorch model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_coarse_expert.model.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for ACOTPytorch model")

    def is_gradient_checkpointing_enabled(self):
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        return torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time = sample_beta(1.5, 1.0, bsize, device) * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_language_tokens, lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(
            pad_masks.shape[0], -1
        )
        return embs, pad_masks, att_masks

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        *,
        explicit_action_reason: Tensor | None = None,
        implicit_action_reason: Tensor | None = None,
        suf_type: Literal["reasoner", "expert"] = "expert",
    ):
        embs = []
        pad_masks = []
        att_masks = []
        if not self.pi05:
            state_emb = self._apply_checkpoint(self.state_proj, state)
            embs.append(state_emb[:, None, :])
            pad_masks.append(torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state.device))
            att_masks += [1]

        if suf_type == "reasoner":
            action_in_proj = self.coarse_action_in_proj
            horizon = self.coarse_action_horizon
        else:
            action_in_proj = self.action_in_proj
            horizon = self.action_horizon

        action_emb = self._apply_checkpoint(action_in_proj, noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        ).type(dtype=timestep.dtype)

        if self.pi05:
            if suf_type == "reasoner":
                time_emb = self.coarse_time_mlp_out(F.silu(self.coarse_time_mlp_in(time_emb)))
            else:
                time_emb = self.time_mlp_out(F.silu(self.time_mlp_in(time_emb)))
            action_expert_tokens = action_emb
            adarms_cond = F.silu(time_emb)
        else:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
            if suf_type == "reasoner":
                action_expert_tokens = self.coarse_action_time_mlp_out(
                    F.silu(self.coarse_action_time_mlp_in(action_time_emb))
                )
            else:
                action_expert_tokens = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))
            adarms_cond = None

        if suf_type == "expert":
            action_expert_tokens = self._apply_reasoning(
                action_expert_tokens,
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
            )

        embs.append(action_expert_tokens)
        pad_masks.append(torch.ones(action_expert_tokens.shape[:2], dtype=torch.bool, device=noisy_actions.device))
        att_masks += [1] + ([0] * (horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)[None, :].expand(embs.shape[0], -1)
        return embs, pad_masks, att_masks, adarms_cond

    def _apply_reasoning(
        self,
        action_expert_tokens: Tensor,
        *,
        explicit_action_reason: Tensor | None,
        implicit_action_reason: Tensor | None,
    ) -> Tensor:
        if self.adopt_explicit_action_reasoner and self.adopt_implicit_action_reasoner:
            explicit_tokens = self.coarse_action_in_proj(explicit_action_reason)
            aligned_explicit = self.explicit_action_reasoner(action_expert_tokens, explicit_tokens)
            aligned_implicit = self.implicit_action_reasoner_interact(action_expert_tokens, implicit_action_reason)
            explicit_branch = self.explicit_action_reason_proj(
                torch.cat([action_expert_tokens, aligned_explicit], dim=-1)
            )
            implicit_branch = self.implicit_action_reason_proj(
                torch.cat([action_expert_tokens, aligned_implicit], dim=-1)
            )
            fused = torch.cat([explicit_branch, implicit_branch], dim=-1)
            return self.action_reasoning_fusion(fused, fused)
        if self.adopt_explicit_action_reasoner:
            explicit_tokens = self.coarse_action_in_proj(explicit_action_reason)
            aligned_explicit = self.explicit_action_reasoner(action_expert_tokens, explicit_tokens)
            return self.action_reasoning_fusion(torch.cat([action_expert_tokens, aligned_explicit], dim=-1))
        if self.adopt_implicit_action_reasoner:
            aligned_implicit = self.implicit_action_reasoner_interact(action_expert_tokens, implicit_action_reason)
            return self.action_reasoning_fusion(torch.cat([action_expert_tokens, aligned_implicit], dim=-1))
        return action_expert_tokens

    def _kv_cache_to_layer_tokens(self, past_key_values):
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        keys = []
        values = []
        for layer_cache in past_key_values:
            key, value = layer_cache[:2]
            # [B, H, T, D] -> [B, T, D], averaging KV heads for the compact reasoner.
            keys.append(key.mean(dim=1))
            values.append(value.mean(dim=1))
        return torch.stack(keys, dim=1).float(), torch.stack(values, dim=1).float()

    def _prefix_cache_and_reason(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        if not self.adopt_implicit_action_reasoner:
            return past_key_values, None
        keys, values = self._kv_cache_to_layer_tokens(past_key_values)
        return past_key_values, self.implicit_action_reasoner(keys, values)

    def _full_forward(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        adarms_cond,
        stream_idx: int,
    ):
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        inputs = [prefix_embs, None, None]
        inputs[stream_idx] = suffix_embs
        adarms = [None, None, None]
        adarms[stream_idx] = adarms_cond
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs,
            use_cache=False,
            adarms_cond=adarms,
        )
        return outputs[stream_idx]

    def forward(self, observation, actions, coarse_actions=None, noise=None, coarse_noise=None, time=None) -> Tensor:
        if coarse_actions is None:
            coarse_actions = actions[:, : self.coarse_action_horizon]
            if coarse_actions.shape[1] != self.coarse_action_horizon:
                raise ValueError("coarse_actions must be provided when action_horizon is shorter than coarse horizon.")

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if coarse_noise is None:
            coarse_noise = self.sample_noise(coarse_actions.shape, coarse_actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_expert_t = time_expanded * noise + (1 - time_expanded) * actions
        u_expert_t = noise - actions
        x_ref_t = time_expanded * coarse_noise + (1 - time_expanded) * coarse_actions
        u_ref_t = coarse_noise - coarse_actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_q_proj = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj
        if prefix_q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        losses = []
        explicit_action_reason = None
        if self.adopt_explicit_action_reasoner:
            suffix_ref_embs, suffix_ref_pad_masks, suffix_ref_att_masks, adarms_ref_cond = self.embed_suffix(
                state, x_ref_t, time, suf_type="reasoner"
            )
            if prefix_embs.dtype == torch.bfloat16:
                suffix_ref_embs = suffix_ref_embs.to(dtype=torch.bfloat16)
            suffix_ref_out = self._full_forward(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                suffix_ref_embs,
                suffix_ref_pad_masks,
                suffix_ref_att_masks,
                adarms_ref_cond,
                stream_idx=1,
            )
            v_ref_t = self.coarse_action_out_proj(suffix_ref_out[:, -self.coarse_action_horizon :].float())
            losses.append(F.mse_loss(u_ref_t, v_ref_t))
            explicit_action_reason = coarse_actions

        implicit_action_reason = None
        if self.adopt_implicit_action_reasoner:
            _, implicit_action_reason = self._prefix_cache_and_reason(prefix_embs, prefix_pad_masks, prefix_att_masks)
        suffix_expert_embs, suffix_expert_pad_masks, suffix_expert_att_masks, adarms_expert_cond = self.embed_suffix(
            state,
            x_expert_t,
            time,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suf_type="expert",
        )
        if prefix_embs.dtype == torch.bfloat16:
            suffix_expert_embs = suffix_expert_embs.to(dtype=torch.bfloat16)
        suffix_expert_out = self._full_forward(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            suffix_expert_embs,
            suffix_expert_pad_masks,
            suffix_expert_att_masks,
            adarms_expert_cond,
            stream_idx=2,
        )
        v_expert_t = self.action_out_proj(suffix_expert_out[:, -self.action_horizon :].float())
        losses.append(F.mse_loss(u_expert_t, v_expert_t))
        return torch.stack(losses).sum()

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, coarse_noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        device = torch.device(device)
        if noise is None:
            noise = self.sample_noise((bsize, self.action_horizon, self.action_dim), device)
        if coarse_noise is None:
            coarse_noise = self.sample_noise((bsize, self.coarse_action_horizon, self.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        past_key_values, implicit_action_reason = self._prefix_cache_and_reason(
            prefix_embs, prefix_pad_masks, prefix_att_masks
        )

        explicit_action_reason = None
        if self.adopt_explicit_action_reasoner:
            explicit_action_reason = self._denoise(
                state,
                prefix_pad_masks,
                past_key_values,
                coarse_noise,
                num_steps,
                suf_type="reasoner",
            )

        actions = self._denoise(
            state,
            prefix_pad_masks,
            past_key_values,
            noise,
            num_steps,
            suf_type="expert",
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
        )
        if self.adopt_explicit_action_reasoner:
            return {"actions": actions, "coarse_actions": explicit_action_reason}
        return {"actions": actions}

    def _denoise(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        num_steps: int,
        *,
        suf_type: Literal["reasoner", "expert"],
        explicit_action_reason: Tensor | None = None,
        implicit_action_reason: Tensor | None = None,
    ):
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=x_t.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=x_t.device)
        while time >= -dt / 2:
            timestep = time.expand(x_t.shape[0])
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
                suf_type=suf_type,
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        *,
        suf_type: Literal["reasoner", "expert"],
        explicit_action_reason: Tensor | None = None,
        implicit_action_reason: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state,
            x_t,
            timestep,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suf_type=suf_type,
        )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

        stream_idx = 1 if suf_type == "reasoner" else 2
        expert = (
            self.paligemma_with_expert.gemma_coarse_expert.model
            if stream_idx == 1
            else self.paligemma_with_expert.gemma_expert.model
        )
        expert.config._attn_implementation = "eager"  # noqa: SLF001
        inputs = [None, None, None]
        inputs[stream_idx] = suffix_embs
        adarms = [None, None, None]
        adarms[stream_idx] = adarms_cond
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs,
            use_cache=False,
            adarms_cond=adarms,
        )
        suffix_out = outputs[stream_idx].float()
        if suf_type == "reasoner":
            return self.coarse_action_out_proj(suffix_out[:, -self.coarse_action_horizon :])
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])
