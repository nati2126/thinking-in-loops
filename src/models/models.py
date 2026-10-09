"""The Universal Transformer and the two baselines, behind one common interface.

Every model exposes the same recurrent API so training and evaluation code are
shared:

    e = model.embed(puzzles)            # (B, 16, d) puzzle embedding
    h = model.init_hidden(e)            # initial hidden state
    h = model.step(h, e)                # one "loop" (UT) or the whole network (baselines)
    logits, halt = model.readout(h)     # (B, 16, 4) digit logits, (B,) halting logit or None

Baselines run exactly one step; the UT runs ``n_loops`` steps of one shared block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig
from ..data.sudoku import SIZE
from .layers import RMSNorm, SudokuEmbedding, TransformerBlock, init_linear_weights


@dataclass
class ModelOutput:
    """Outputs for every loop.

    Attributes:
        logits: (L, B, 16, 4) digit logits after each of the L loops (class k = digit k+1).
        halt_logits: (L, B) halting logits (pre-sigmoid) or None for models without a halting head.
    """

    logits: torch.Tensor
    halt_logits: torch.Tensor | None


class SudokuModel(nn.Module):
    """Shared embedding + readout. Subclasses define ``init_hidden`` and ``step``."""

    is_recurrent: bool = False
    has_halting: bool = False

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = SudokuEmbedding(cfg.d_model)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, SIZE)  # shared output head, digits 1..4
        self.halt_head = nn.Linear(cfg.d_model, 1) if self.has_halting else None

    @property
    def default_loops(self) -> int:
        return 1

    def embed(self, puzzles: torch.Tensor) -> torch.Tensor:
        return self.embedding(puzzles)

    def init_hidden(self, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def readout(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.final_norm(h)
        logits = self.head(z)
        halt = self.halt_head(z.mean(dim=1)).squeeze(-1) if self.halt_head is not None else None
        return logits, halt

    def forward(self, puzzles: torch.Tensor, n_loops: int | None = None) -> ModelOutput:
        """Run ``n_loops`` steps (ignored by non-recurrent baselines) and return every loop's output."""
        n_loops = n_loops if (self.is_recurrent and n_loops is not None) else self.default_loops
        e = self.embed(puzzles)
        h = self.init_hidden(e)
        all_logits, all_halts = [], []
        for _ in range(n_loops):
            h = self.step(h, e)
            logits, halt = self.readout(h)
            all_logits.append(logits)
            if halt is not None:
                all_halts.append(halt)
        return ModelOutput(torch.stack(all_logits), torch.stack(all_halts) if all_halts else None)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class UniversalTransformer(SudokuModel):
    """ONE transformer block applied repeatedly, with input injection and a halting head.

    h_0 = 0;  h_{t+1} = Block(h_t + e)  where e is the puzzle embedding.
    With h_0 = 0, loop 1 computes exactly what a 1-layer transformer would.
    """

    is_recurrent = True
    has_halting = True

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.block = TransformerBlock(cfg)
        init_linear_weights(self)

    @property
    def default_loops(self) -> int:
        return self.cfg.max_loops_train

    def init_hidden(self, e: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(e)

    def step(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.block(h + e)


class StackedTransformer(SudokuModel):
    """Standard (non-looped) transformer with ``n_layers`` distinct blocks."""

    def __init__(self, cfg: ModelConfig, n_layers: int) -> None:
        super().__init__(cfg)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(n_layers))
        init_linear_weights(self)

    def init_hidden(self, e: torch.Tensor) -> torch.Tensor:
        return e

    def step(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h


MODEL_NAMES = ("ut", "same_param", "same_compute")


def build_model(name: str, cfg: ModelConfig) -> SudokuModel:
    """Build a model by name.

    * ``ut``           - Universal Transformer (1 shared block, looped).
    * ``same_param``   - 1-layer standard transformer (same parameters as the UT, 1/8 compute).
    * ``same_compute`` - 8 distinct layers (same FLOPs as 8 UT loops, ~8x parameters).
    """
    if name == "ut":
        return UniversalTransformer(cfg)
    if name == "same_param":
        return StackedTransformer(cfg, n_layers=1)
    if name == "same_compute":
        return StackedTransformer(cfg, n_layers=cfg.same_compute_layers)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
