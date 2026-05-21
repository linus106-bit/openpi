import dataclasses
from collections.abc import Mapping

import torch


COARSE_EXPERT_PREFIX = "paligemma_with_expert.gemma_experts.0."
FINAL_EXPERT_PREFIX = "paligemma_with_expert.gemma_experts.1."

_ACOT_RANDOM_INIT_PREFIXES = (
    "explicit_action_reasoner.",
    "implicit_action_reasoner.",
    "implicit_action_reasoner_interact.",
    "explicit_action_reason_proj.",
    "implicit_action_reason_proj.",
    "action_reasoning_fusion.",
)


@dataclasses.dataclass(frozen=True)
class ACOTExpertCloneResult:
    state_dict: dict[str, torch.Tensor]
    cloned_destination_keys: frozenset[str]
    skipped_destination_keys: frozenset[str]


def clone_acot_final_expert_from_coarse(
    state_dict: Mapping[str, torch.Tensor],
    target_state_dict: Mapping[str, torch.Tensor],
) -> ACOTExpertCloneResult:
    """Clone ACOT final expert weights from coarse expert weights when target shapes match."""
    expanded = dict(state_dict)
    cloned_destination_keys: set[str] = set()
    skipped_destination_keys: set[str] = set()

    for destination_key, target_tensor in target_state_dict.items():
        if not destination_key.startswith(FINAL_EXPERT_PREFIX):
            continue

        suffix = destination_key.removeprefix(FINAL_EXPERT_PREFIX)
        source_key = f"{COARSE_EXPERT_PREFIX}{suffix}"
        source_tensor = expanded.get(source_key)

        if destination_key in expanded or source_tensor is None:
            skipped_destination_keys.add(destination_key)
            continue

        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            skipped_destination_keys.add(destination_key)
            continue

        expanded[destination_key] = source_tensor.to(dtype=target_tensor.dtype)
        cloned_destination_keys.add(destination_key)

    return ACOTExpertCloneResult(
        state_dict=expanded,
        cloned_destination_keys=frozenset(cloned_destination_keys),
        skipped_destination_keys=frozenset(skipped_destination_keys),
    )


def is_expected_missing_acot_key(key: str, *, skipped_final_expert_keys: frozenset[str]) -> bool:
    """Return whether a missing ACOT key is expected after PI0/PI05 checkpoint bootstrapping."""
    return key in skipped_final_expert_keys or key.startswith(_ACOT_RANDOM_INIT_PREFIXES)
