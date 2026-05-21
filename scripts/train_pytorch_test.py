import dataclasses
import pathlib
import types

import safetensors.torch
import torch

import train_pytorch


@dataclasses.dataclass
class _CheckpointConfig:
    checkpoint_dir: pathlib.Path
    ema_decay: float | None
    save_interval: int = 1
    num_train_steps: int = 10
    wandb_enabled: bool = False


def _data_config():
    return types.SimpleNamespace(norm_stats=None, asset_id=None)


def _linear(weight: float):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def _weight(model):
    return model.weight.detach().clone()


def _save_ema_checkpoint(tmp_path, *, raw_weight: float, ema_weight: float, decay: float = 0.9):
    model = _linear(raw_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = train_pytorch.create_ema_model(model, decay, torch.device("cpu"))
    with torch.no_grad():
        train_pytorch.ema_module(ema_model).weight.fill_(ema_weight)
    train_pytorch.set_ema_n_averaged(ema_model, 4)

    config = _CheckpointConfig(checkpoint_dir=tmp_path, ema_decay=decay)
    train_pytorch.save_checkpoint(model, optimizer, 1, config, True, _data_config(), ema_model=ema_model)
    return tmp_path / "1"


def test_ema_first_update_matches_jax_formula():
    model = _linear(1.0)
    ema_model = train_pytorch.create_ema_model(model, 0.9, torch.device("cpu"))

    with torch.no_grad():
        model.weight.fill_(3.0)
    ema_model.update_parameters(model)

    torch.testing.assert_close(_weight(train_pytorch.ema_module(ema_model)), torch.tensor([[1.2]]))
    assert int(ema_model.n_averaged.item()) == 2


def test_ema_checkpoint_saves_ema_for_inference_and_raw_for_resume(tmp_path):
    ckpt_dir = _save_ema_checkpoint(tmp_path, raw_weight=5.0, ema_weight=2.0)

    inference_state = safetensors.torch.load_file(ckpt_dir / "model.safetensors")
    training_state = safetensors.torch.load_file(ckpt_dir / "training_model.safetensors")

    assert set(inference_state) == {"weight"}
    assert set(training_state) == {"weight"}
    torch.testing.assert_close(inference_state["weight"], torch.tensor([[2.0]]))
    torch.testing.assert_close(training_state["weight"], torch.tensor([[5.0]]))


def test_load_checkpoint_prefers_training_weights_when_ema_disabled(tmp_path):
    _save_ema_checkpoint(tmp_path, raw_weight=5.0, ema_weight=2.0)
    model = _linear(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    step = train_pytorch.load_checkpoint(model, optimizer, tmp_path, torch.device("cpu"), ema_model=None)

    assert step == 1
    torch.testing.assert_close(_weight(model), torch.tensor([[5.0]]))


def test_load_checkpoint_restores_ema_shadow(tmp_path):
    _save_ema_checkpoint(tmp_path, raw_weight=5.0, ema_weight=2.0, decay=0.9)
    model = _linear(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = train_pytorch.create_ema_model(model, 0.9, torch.device("cpu"))

    step = train_pytorch.load_checkpoint(
        model,
        optimizer,
        tmp_path,
        torch.device("cpu"),
        ema_model=ema_model,
        config_ema_decay=0.9,
    )

    assert step == 1
    torch.testing.assert_close(_weight(model), torch.tensor([[5.0]]))
    torch.testing.assert_close(_weight(train_pytorch.ema_module(ema_model)), torch.tensor([[2.0]]))
    assert int(ema_model.n_averaged.item()) == 4


def test_load_checkpoint_resets_ema_on_decay_mismatch(tmp_path):
    _save_ema_checkpoint(tmp_path, raw_weight=5.0, ema_weight=2.0, decay=0.9)
    model = _linear(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema_model = train_pytorch.create_ema_model(model, 0.8, torch.device("cpu"))

    train_pytorch.load_checkpoint(
        model,
        optimizer,
        tmp_path,
        torch.device("cpu"),
        ema_model=ema_model,
        config_ema_decay=0.8,
    )

    torch.testing.assert_close(_weight(model), torch.tensor([[5.0]]))
    torch.testing.assert_close(_weight(train_pytorch.ema_module(ema_model)), torch.tensor([[5.0]]))
    assert int(ema_model.n_averaged.item()) == 1


def test_no_ema_checkpoint_keeps_legacy_single_model_file(tmp_path):
    model = _linear(4.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _CheckpointConfig(checkpoint_dir=tmp_path, ema_decay=None)

    train_pytorch.save_checkpoint(model, optimizer, 1, config, True, _data_config(), ema_model=None)

    ckpt_dir = tmp_path / "1"
    assert (ckpt_dir / "model.safetensors").exists()
    assert not (ckpt_dir / "training_model.safetensors").exists()

    restored = _linear(0.0)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    train_pytorch.load_checkpoint(restored, restored_optimizer, tmp_path, torch.device("cpu"), ema_model=None)
    torch.testing.assert_close(_weight(restored), torch.tensor([[4.0]]))
