"""Central configuration for data, models and training.

Every tunable number lives here as a dataclass default. Scripts expose these
fields as command-line flags via :func:`add_dataclass_args`, so nothing is
hard-coded elsewhere.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Puzzle generation and dataset split settings."""

    data_dir: str = str(REPO_ROOT / "data")
    seed: int = 1234
    min_empty: int = 4
    max_empty: int = 8
    n_train: int = 20_000
    n_val: int = 2_000
    n_test: int = 2_000


@dataclass
class ModelConfig:
    """Architecture hyper-parameters shared by the UT and the baselines."""

    d_model: int = 128
    n_heads: int = 4
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * d_model (rounded)
    mlp_multiple_of: int = 32
    dropout: float = 0.0
    max_loops_train: int = 8  # UT loops during training
    same_compute_layers: int = 8  # depth of the unshared "same-compute" baseline
    norm_eps: float = 1e-6


@dataclass
class TrainConfig:
    """Optimisation and logging settings."""

    model: str = "ut"  # one of: ut, same_param, same_compute
    run_name: str = ""  # defaults to the model name
    seed: int = 0
    epochs: int = 40
    batch_size: int = 256
    eval_batch_size: int = 1024
    lr: float = 1e-3
    min_lr_ratio: float = 0.05  # cosine decays to lr * min_lr_ratio
    warmup_frac: float = 0.05
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    tbptt_k: int = 3  # gradients flow through at most K loops; 0 = all loops
    halt_loss_weight: float = 0.5
    augment: bool = True
    amp: bool = True  # bf16 autocast on CUDA; ignored on CPU
    halt_threshold: float = 0.5
    resume: bool = False
    checkpoint_dir: str = str(REPO_ROOT / "checkpoints")
    results_dir: str = str(REPO_ROOT / "results")
    device: str = "auto"

    def resolved_run_name(self) -> str:
        return self.run_name or self.model


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type, prefix: str = "") -> None:
    """Register every field of a dataclass as a ``--flag`` on ``parser``."""
    for f in dataclasses.fields(cls):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        name = f"--{prefix}{f.name}"
        if isinstance(default, bool):
            parser.add_argument(name, type=_str2bool, default=default, help=f"(default: {default})")
        elif isinstance(default, tuple):
            parser.add_argument(name, type=type(default[0]), nargs=len(default), default=default)
        else:
            parser.add_argument(name, type=type(default), default=default, help=f"(default: {default})")


def dataclass_from_args(cls: type, args: argparse.Namespace, prefix: str = ""):
    """Build a dataclass instance from parsed args registered with the same prefix."""
    kwargs = {}
    for f in dataclasses.fields(cls):
        value = getattr(args, f"{prefix}{f.name}")
        kwargs[f.name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "y"):
        return True
    if v.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def resolve_device(device: str = "auto"):
    """Return ``cuda`` when available (or requested), else ``cpu``."""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "REPO_ROOT",
    "add_dataclass_args",
    "dataclass_from_args",
    "resolve_device",
    "set_seed",
]
