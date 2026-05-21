import torch

from openpi.models_pytorch import acot_checkpoint


def test_clone_acot_final_expert_from_coarse_copies_matching_target_shape_and_dtype():
    source_key = f"{acot_checkpoint.COARSE_EXPERT_PREFIX}model.layers.0.self_attn.q_proj.weight"
    destination_key = f"{acot_checkpoint.FINAL_EXPERT_PREFIX}model.layers.0.self_attn.q_proj.weight"
    state_dict = {source_key: torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    target_state_dict = {destination_key: torch.empty((1, 2), dtype=torch.bfloat16)}

    result = acot_checkpoint.clone_acot_final_expert_from_coarse(state_dict, target_state_dict)

    assert result.cloned_destination_keys == frozenset({destination_key})
    assert result.skipped_destination_keys == frozenset()
    assert result.state_dict[destination_key].dtype == torch.bfloat16
    torch.testing.assert_close(result.state_dict[destination_key].float(), state_dict[source_key])


def test_clone_acot_final_expert_from_coarse_does_not_overwrite_existing_destination():
    suffix = "model.layers.0.mlp.down_proj.weight"
    source_key = f"{acot_checkpoint.COARSE_EXPERT_PREFIX}{suffix}"
    destination_key = f"{acot_checkpoint.FINAL_EXPERT_PREFIX}{suffix}"
    original_destination = torch.full((2, 2), 9.0)
    state_dict = {
        source_key: torch.ones((2, 2)),
        destination_key: original_destination,
    }
    target_state_dict = {destination_key: torch.empty((2, 2))}

    result = acot_checkpoint.clone_acot_final_expert_from_coarse(state_dict, target_state_dict)

    assert result.cloned_destination_keys == frozenset()
    assert result.skipped_destination_keys == frozenset({destination_key})
    assert result.state_dict[destination_key] is original_destination


def test_clone_acot_final_expert_from_coarse_skips_shape_mismatch():
    suffix = "model.layers.0.mlp.gate_proj.weight"
    source_key = f"{acot_checkpoint.COARSE_EXPERT_PREFIX}{suffix}"
    destination_key = f"{acot_checkpoint.FINAL_EXPERT_PREFIX}{suffix}"
    state_dict = {source_key: torch.ones((2, 2))}
    target_state_dict = {destination_key: torch.empty((3, 2))}

    result = acot_checkpoint.clone_acot_final_expert_from_coarse(state_dict, target_state_dict)

    assert result.cloned_destination_keys == frozenset()
    assert result.skipped_destination_keys == frozenset({destination_key})
    assert destination_key not in result.state_dict


def test_is_expected_missing_acot_key_uses_exact_skipped_final_expert_allowlist():
    skipped_key = f"{acot_checkpoint.FINAL_EXPERT_PREFIX}model.layers.0.mlp.up_proj.weight"
    unrelated_final_key = f"{acot_checkpoint.FINAL_EXPERT_PREFIX}model.layers.1.mlp.up_proj.weight"

    assert acot_checkpoint.is_expected_missing_acot_key(
        skipped_key,
        skipped_final_expert_keys=frozenset({skipped_key}),
    )
    assert not acot_checkpoint.is_expected_missing_acot_key(
        unrelated_final_key,
        skipped_final_expert_keys=frozenset({skipped_key}),
    )
    assert acot_checkpoint.is_expected_missing_acot_key(
        "explicit_action_reasoner.attention.weight",
        skipped_final_expert_keys=frozenset(),
    )
    assert not acot_checkpoint.is_expected_missing_acot_key(
        "action_out_proj.weight",
        skipped_final_expert_keys=frozenset(),
    )
