from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()
        expert_configs = (
            list(action_expert_config)
            if isinstance(action_expert_config, Sequence) and not isinstance(action_expert_config, str)
            else [action_expert_config]
        )
        if use_adarms is None:
            use_adarms = [False] * (len(expert_configs) + 1)
        if len(use_adarms) != len(expert_configs) + 1:
            raise ValueError("use_adarms must include one flag for PaliGemma and one per expert.")

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
        self.gemma_experts = nn.ModuleList(
            self._make_expert(config, use_adarms[idx + 1]) for idx, config in enumerate(expert_configs)
        )
        for expert in self.gemma_experts:
            expert.model.embed_tokens = None

        # Backward-compatible alias used by the PI0 PyTorch model.
        self.gemma_expert = self.gemma_experts[0]
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
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

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

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _stream_models(self):
        return [self.paligemma.language_model, *[expert.model for expert in self.gemma_experts]]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        if len(inputs_embeds) != len(self.gemma_experts) + 1:
            raise ValueError(f"Expected {len(self.gemma_experts) + 1} input streams, got {len(inputs_embeds)}.")
        if adarms_cond is None:
            adarms_cond = [None] * len(inputs_embeds)

        active_indices = [idx for idx, embeds in enumerate(inputs_embeds) if embeds is not None]
        if not active_indices:
            raise ValueError("At least one input stream must be provided.")

        if active_indices == [0]:
            output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0],
            )
            outputs = [None] * len(inputs_embeds)
            outputs[0] = output.last_hidden_state
            return outputs, output.past_key_values

        if len(active_indices) == 1:
            stream_idx = active_indices[0]
            expert = self.gemma_experts[stream_idx - 1].model
            output = expert.forward(
                inputs_embeds=inputs_embeds[stream_idx],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[stream_idx],
            )
            outputs = [None] * len(inputs_embeds)
            outputs[stream_idx] = output.last_hidden_state
            return outputs, output.past_key_values

        return self._forward_joint(attention_mask, position_ids, inputs_embeds, adarms_cond, active_indices), None

    def _forward_joint(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: list[torch.Tensor | None],
        adarms_cond: list[torch.Tensor | None],
        active_indices: list[int],
    ):
        models = self._stream_models()
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        hidden_states_by_stream = list(inputs_embeds)
        use_gradient_checkpointing = self.training and any(
            getattr(models[stream_idx], "gradient_checkpointing", False) for stream_idx in active_indices
        )

        def compute_layer_complete(layer_idx, hidden_states_by_stream, attention_mask, position_ids, adarms_cond):
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
            return next_hidden_states

        def checkpointed_layer(layer_idx, hidden_states_by_stream, attention_mask, position_ids, adarms_cond):
            active_hidden_states = tuple(hidden_states_by_stream[stream_idx] for stream_idx in active_indices)
            adarms_tensor_indices = [stream_idx for stream_idx in active_indices if adarms_cond[stream_idx] is not None]
            active_adarms = tuple(adarms_cond[stream_idx] for stream_idx in adarms_tensor_indices)

            def checkpoint_func(*tensor_args):
                split = len(active_indices)
                checkpoint_hidden_states = list(hidden_states_by_stream)
                checkpoint_adarms = list(adarms_cond)
                for idx, stream_idx in enumerate(active_indices):
                    checkpoint_hidden_states[stream_idx] = tensor_args[idx]
                for idx, stream_idx in enumerate(adarms_tensor_indices):
                    checkpoint_adarms[stream_idx] = tensor_args[split + idx]
                next_states = compute_layer_complete(
                    layer_idx, checkpoint_hidden_states, attention_mask, position_ids, checkpoint_adarms
                )
                return tuple(next_states[stream_idx] for stream_idx in active_indices)

            checkpoint_args = (*active_hidden_states, *active_adarms)
            next_active_hidden_states = torch.utils.checkpoint.checkpoint(
                checkpoint_func, *checkpoint_args, use_reentrant=False, preserve_rng_state=False
            )
            if not isinstance(next_active_hidden_states, tuple):
                next_active_hidden_states = (next_active_hidden_states,)
            next_hidden_states = list(hidden_states_by_stream)
            for stream_idx, hidden_state in zip(active_indices, next_active_hidden_states, strict=True):
                next_hidden_states[stream_idx] = hidden_state
            return next_hidden_states

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                hidden_states_by_stream = checkpointed_layer(
                    layer_idx, hidden_states_by_stream, attention_mask, position_ids, adarms_cond
                )
            else:
                hidden_states_by_stream = compute_layer_complete(
                    layer_idx, hidden_states_by_stream, attention_mask, position_ids, adarms_cond
                )

        outputs = [None] * len(inputs_embeds)
        for stream_idx in active_indices:
            out_emb, _ = models[stream_idx].norm(hidden_states_by_stream[stream_idx], cond=adarms_cond[stream_idx])
            outputs[stream_idx] = out_emb
        return outputs
