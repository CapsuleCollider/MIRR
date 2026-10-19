"""Gated diagonal selective state space module.

The long-horizon encoder of MIRR, T^2ARec, TTT4Rec and DT3OR is a gated diagonal
selective SSM (LRU-style):

    a_t = sigmoid(W_a x_t)          # input-dependent decay (the selectivity gate)
    u_t = W_u x_t ,  v_t = W_v x_t
    w_t = (1 - a_t) * u_t
    h_t = a_t * h_{t-1} + w_t       # diagonal state recurrence
    y_t = W_o ( h_t * SiLU(v_t) )   # output projection with gating

Two properties matter for this repository:

* **Training** uses a closed-form parallel scan (``cumprod``/``cumsum``), which is
  numerically identical to stepping the recurrence, so a whole sequence is encoded
  in one forward pass.
* **Serving** uses :meth:`SSMBlock.apply` one interaction at a time, which is what
  makes the "fixed-size state, constant cost per interaction" claim of the paper
  concrete.

Note. T^2ARec uses Mamba-2 as its sequence backbone. ``mamba-ssm`` requires a local
CUDA build on Windows and does not support the step-by-step state update this
repository needs, so we use a same-family block that is trainable in parallel and
steppable online. This is stated in the paper appendix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_FLOOR = -13.8  # exp(-13.8) ~ 1e-6: prevents underflow of the cumprod on long sequences


class SSMBlock(nn.Module):
    """Pre-LayerNorm + selective SSM + residual + FFN."""

    def __init__(self, d_model: int, d_state: int | None = None, expand: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = D = d_state or d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * D, bias=False)   # -> (u, v)
        self.decay = nn.Linear(d_model, D)                     # -> a
        self.out_proj = nn.Linear(D, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

        nn.init.constant_(self.decay.bias, 2.2)                # initial a ~ 0.9
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    # ------------------------------------------------------------------ #
    # Coefficients. Any leading dimensions are allowed, which is what lets a
    # single block serve K parallel slots through broadcasting.
    # ------------------------------------------------------------------ #
    def coeffs(self, x: torch.Tensor):
        """x: (..., d) -> a (..., D), w (..., D), v (..., D)."""
        u, v = self.in_proj(x).chunk(2, dim=-1)
        a = torch.sigmoid(self.decay(x))
        return a, (1.0 - a) * u, v

    def apply(self, h: torch.Tensor, a: torch.Tensor, w: torch.Tensor, v: torch.Tensor):
        """One recurrence step. h: (..., D) -> y: (..., d), h_new: (..., D)."""
        h_new = a * h + w
        return self.out_proj(h_new * F.silu(v)), h_new

    # ------------------------------------------------------------------ #
    # Parallel (training)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x: (B, T, d) -> y: (B, T, d), h_last: (B, D)."""
        a, w, v = self.coeffs(self.norm1(x))                   # (B, T, D)
        log_a = torch.log(a.clamp(min=1e-4))
        logP = torch.cumsum(log_a, dim=1).clamp(min=LOG_FLOOR)
        P = torch.exp(logP)
        S = torch.cumsum(w * torch.exp(-logP), dim=1)
        h = P * S
        if state is not None:                                  # continue from a state
            h = h + P * state.unsqueeze(1)
        y = self.out_proj(h * F.silu(v))
        x = x + self.drop(y)
        x = x + self.ffn(self.norm2(x))
        return x, h[:, -1]

    # ------------------------------------------------------------------ #
    # Stepwise (serving / streaming evaluation)
    # ------------------------------------------------------------------ #
    def step(self, x_t: torch.Tensor, h: torch.Tensor):
        """x_t: (..., d), h: (..., D) -> out: (..., d), h_new: (..., D)."""
        a, w, v = self.coeffs(self.norm1(x_t))
        y, h_new = self.apply(h, a, w, v)
        o = x_t + y
        return o + self.ffn(self.norm2(o)), h_new

    def init_state(self, B, K, device, dtype=torch.float32):
        return torch.zeros(B, K, self.d_state, device=device, dtype=dtype)
