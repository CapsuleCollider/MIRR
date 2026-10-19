"""Shared components: item embedding, ranking metrics, and the model interface.

Every method in this repository implements :class:`SeqRecModel`, so the training
engine and the evaluator are method-agnostic.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class ItemEmbedding(nn.Module):
    """Item embedding plus a learnable positional embedding.

    Index ``0`` is reserved for padding and is kept at exactly zero.
    """

    def __init__(self, n_items: int, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.item = nn.Embedding(n_items, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].fill_(0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T) -> (B, T, d)."""
        T = seq.size(1)
        pos = torch.arange(T, device=seq.device).unsqueeze(0)
        return self.drop(self.norm(self.item(seq) + self.pos(pos)))


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, ks=(10, 20)):
    """Full-catalog ranking metrics.

    Args:
        logits: (B, n_items) score of every catalog item.
        target: (B,) ground-truth next item.

    Returns:
        Recall@K / NDCG@K for every K in ``ks``, plus MRR. The rank is 0-based and
        computed against the *whole* catalog, so no negative sampling is involved.
    """
    tgt_score = logits.gather(1, target.unsqueeze(1))          # (B, 1)
    rank = (logits > tgt_score).sum(dim=1)                     # (B,) 0-based rank
    out = {}
    for k in ks:
        hit = (rank < k).float()
        out[f"Recall@{k}"] = hit.mean().item()
        out[f"NDCG@{k}"] = (hit / torch.log2(rank.float() + 2.0)).mean().item()
    out["MRR"] = (1.0 / (rank.float() + 1.0)).mean().item()
    return out


def mean_metrics(dicts):
    """Average a list of metric dicts (used to aggregate over batches / seeds)."""
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def paired_t(deltas):
    """Two-sided paired t statistic and 5%/10% critical values for ``len(deltas)`` seeds.

    With 3 seeds the degrees of freedom are 2, giving t_{0.05} = 4.303 and
    t_{0.10} = 2.920. Used by ``scripts/make_tables.py`` to mark significance.
    """
    x = np.asarray(deltas, dtype=float)
    n = len(x)
    if n < 2:
        return dict(n=n, mean=float(x.mean()) if n else float("nan"), t=float("nan"))
    sd = x.std(ddof=1)
    t = float(x.mean() / (sd / math.sqrt(n))) if sd > 0 else float("inf")
    crit = {3: (4.303, 2.920), 4: (3.182, 2.353), 5: (2.776, 2.132)}.get(n)
    return dict(n=n, mean=float(x.mean()), std=float(sd), t=t,
                t_05=crit[0] if crit else None, t_10=crit[1] if crit else None)


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class SeqRecModel(nn.Module):
    """Uniform interface implemented by every method.

    ``forward(seq)``        -> (B, T, n_items): predict the next item at every position.
    ``final_logits(seq)``   -> (B, n_items): prediction at the last valid position.
    ``init_state(B)``       -> recurrent state, for the streaming evaluation protocol.
    ``step(x_t, state)``    -> (logits, new_state): advance the state by one interaction.
    ``ttt_param_groups()``  -> parameters updated at test time (empty = a static model).
    ``ttt_closure(...)``    -> self-supervised loss used by test-time-training methods.
    """

    name = "base"
    is_ttt = False

    def final_logits(self, seq: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def init_state(self, B, device):
        raise NotImplementedError

    def step(self, x_t, state, pos=None, **kw):
        raise NotImplementedError

    def ttt_param_groups(self):
        return []

    def ttt_closure(self, logits, y, state=None):
        return None

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6
