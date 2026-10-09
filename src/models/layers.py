"""Transformer building blocks: embeddings, RMSNorm, attention, SwiGLU, block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from ..data.sudoku import NUM_CELLS, SIZE, box_of, col_of, row_of


class RMSNorm(nn.Module):
    """Root-mean-square layer norm with a learned gain (no bias, no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SudokuEmbedding(nn.Module):
    """Token embedding (0 = empty, 1-4 = digits) + learned row, column and box embeddings."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.token = nn.Embedding(SIZE + 1, d_model)
        self.row = nn.Embedding(SIZE, d_model)
        self.col = nn.Embedding(SIZE, d_model)
        self.box = nn.Embedding(SIZE, d_model)
        cells = range(NUM_CELLS)
        self.register_buffer("row_ids", torch.tensor([row_of(i) for i in cells]), persistent=False)
        self.register_buffer("col_ids", torch.tensor([col_of(i) for i in cells]), persistent=False)
        self.register_buffer("box_ids", torch.tensor([box_of(i) for i in cells]), persistent=False)
        for emb in (self.token, self.row, self.col, self.box):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, puzzles: torch.Tensor) -> torch.Tensor:
        """(B, 16) long -> (B, 16, d_model)."""
        pos = self.row(self.row_ids) + self.col(self.col_ids) + self.box(self.box_ids)
        return self.token(puzzles) + pos


class SelfAttention(nn.Module):
    """Non-causal multi-head self-attention: every cell attends to every cell."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: W_down(SiLU(W_gate x) * W_up x)."""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


def swiglu_hidden(cfg: ModelConfig) -> int:
    """SwiGLU width rounded up to a multiple of ``cfg.mlp_multiple_of``."""
    raw = int(cfg.mlp_ratio * cfg.d_model)
    m = cfg.mlp_multiple_of
    return ((raw + m - 1) // m) * m


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(RMSNorm(x)), then x + SwiGLU(RMSNorm(x))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg.d_model, swiglu_hidden(cfg))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.mlp(self.norm2(x)))


def init_linear_weights(module: nn.Module) -> None:
    """Small normal init for linear layers (GPT-style)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
