"""Saving and loading checkpoints (model weights + the configs needed to rebuild them)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from ..config import ModelConfig
from ..models.models import SudokuModel, build_model


def save_checkpoint(path: Path, model: SudokuModel, model_name: str, extra: dict[str, Any] | None = None) -> None:
    """Write ``model`` weights, its name and config, plus any ``extra`` state (optimizer, epoch, ...)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": model_name, "model_cfg": dataclasses.asdict(model.cfg), "model_state": model.state_dict()}
    payload.update(extra or {})
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on the same filesystem


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def load_model(path: str | Path, device: torch.device | str = "cpu") -> tuple[SudokuModel, dict[str, Any]]:
    """Rebuild a model from a checkpoint. Returns (model in eval mode, raw checkpoint dict)."""
    ckpt = load_checkpoint(path, device)
    model = build_model(ckpt["model_name"], ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
