"""Static baselines: GRU4Rec / SASRec / ComiRec / REMI (lite).

All four are trained offline and frozen at serving time: evaluation is a single
forward pass through :meth:`final_logits` and none of them takes a test-time step.

Fairness note on ComiRec / REMI.
    The conventional implementation supervises a multi-interest model at the last
    position only, i.e. one label per sequence, while SASRec / GRU4Rec and MIRR are
    supervised at *every* position. On 5-core benchmarks, where sequences are short,
    this costs the multi-interest baselines roughly 8x fewer labels and depresses
    them far below what their encoder can do. An earlier remedy was to expand each
    sequence into ``P`` prefixes, but that replicates the same user ``P`` times and
    overfits within a few epochs.

    We instead exploit the causal mask: ``h[:, t]`` is already the encoding of the
    prefix ``0..t``, so masking positions after ``t`` inside the interest-extraction
    attention yields the interest vectors of *every* position from a single forward
    pass -- no duplicated users, no re-encoding. Sampling ``S`` positions per
    sequence (``--n_sup_pos``) is an unbiased estimate of the full-position loss at
    about ``S`` times the cost of the final-layer extraction. This is what
    ``forward_multi_pos`` implements; ``n_sup_pos=8`` is used in the main table.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ItemEmbedding, SeqRecModel


class _Scorer(nn.Module):
    """Output layer tied to the item embedding, with a learnable temperature."""

    def __init__(self, item_emb: nn.Embedding):
        super().__init__()
        self.item_emb = item_emb
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d) or (B, K, d) -> logits (B, n_items) / (B, K, n_items)."""
        E = self.item_emb.weight                     # (n_items, d)
        if h.dim() == 3:
            return torch.einsum("bkd,nd->bkn", h, E) * self.scale
        return (h @ E.T) * self.scale


class GRU4Rec(SeqRecModel):
    name = "GRU4Rec"

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, dropout=0.2):
        super().__init__()
        self.emb = ItemEmbedding(n_items, d_model, max_len, dropout)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.out = _Scorer(self.emb.item)
        self.max_len = max_len

    def forward(self, seq):
        return self.out(self.gru(self.emb(seq))[0])

    def final_logits(self, seq):
        logits = self.forward(seq)                   # (B, T, n_items)
        B = logits.size(0)
        idx = (seq != 0).sum(dim=1) - 1              # last valid position
        return logits[torch.arange(B, device=seq.device), idx.clamp(min=0)]


class SASRec(SeqRecModel):
    name = "SASRec"

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, n_heads=2,
                 dropout=0.2, ffn_mult=4):
        super().__init__()
        self.emb = ItemEmbedding(n_items, d_model, max_len, dropout)
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.out = _Scorer(self.emb.item)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, seq):
        x = self.emb(seq)
        T = x.size(1)
        m = self._causal_mask(T, seq.device)
        pad = (seq == 0)
        for blk in self.blocks:
            x = blk(x, src_mask=m, src_key_padding_mask=pad)
        return self.out(self.ln(x))

    def final_logits(self, seq):
        logits = self.forward(seq)
        B = seq.size(0)
        idx = (seq != 0).sum(dim=1) - 1
        return logits[torch.arange(B, device=seq.device), idx.clamp(min=0)]


class MultiInterestMixin:
    """Interest extraction and diversity regularizer shared by ComiRec-SA and REMI."""

    def _extract(self, hseq, pad_mask, K):
        """Multi-head-attention style extraction. hseq: (B, T, d) -> (B, K, d)."""
        B, T, _ = hseq.shape
        q = self.interest_query.unsqueeze(0).expand(B, -1, -1)          # (B, K, d)
        attn = torch.einsum("btd,bkd->bkt", hseq, q) / math.sqrt(hseq.size(-1))
        attn = attn.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        a = torch.softmax(attn, dim=-1)                                  # (B, K, T)
        self.last_attn = a
        return torch.einsum("bkt,btd->bkd", a, hseq)                     # (B, K, d)

    def _interests_at(self, h, seq, pos):
        """Interest vectors at arbitrary positions. h: (B, T, d), pos: (B, S) -> (B, S, K, d).

        The encoder runs once; only the final extraction is expanded over the ``S``
        sampled supervision positions.
        """
        B, T, d = h.shape
        S = pos.size(1)
        # mask: padding OR positions after the supervision point (causality)
        pad = (seq == 0).unsqueeze(1) | (
            torch.arange(T, device=seq.device).view(1, 1, T) > pos.unsqueeze(2))
        q = self.interest_query.unsqueeze(0).expand(B, -1, -1)          # (B, K, d)
        attn = torch.einsum("btd,bkd->bkt", h, q) / math.sqrt(d)        # (B, K, T)
        attn = attn.unsqueeze(1).masked_fill(pad.unsqueeze(2), float("-inf"))
        a = torch.softmax(attn, dim=-1)                                  # (B, S, K, T)
        return torch.einsum("bskt,btd->bskd", a, h)

    @staticmethod
    def diversity_loss(interests):
        """ComiRec's diversity regularizer: keep the K interest vectors apart."""
        z = F.normalize(interests, dim=-1)                 # (B, K, d)
        sim = torch.einsum("bkd,bld->bkl", z, z)           # (B, K, K)
        eye = torch.eye(interests.size(1), device=z.device).unsqueeze(0)
        return ((sim * (1 - eye)) ** 2).sum() / (
            interests.size(0) * max(1, interests.size(1) - 1))


class ComiRec(SeqRecModel, MultiInterestMixin):
    """ComiRec (self-attentive variant): K interest vectors aggregated by max-pooling."""

    name = "ComiRec"
    train_on_final = True      # supervised at sampled positions (see module docstring)

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, n_heads=2,
                 dropout=0.2, K=4, div_weight=0.01):
        super().__init__()
        self.emb = ItemEmbedding(n_items, d_model, max_len, dropout)
        self.K = K
        self.div_weight = div_weight
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.interest_query = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.out = _Scorer(self.emb.item)

    def _encode(self, seq):
        x = self.emb(seq)
        m = torch.triu(torch.ones(seq.size(1), seq.size(1), dtype=torch.bool,
                                  device=seq.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=m, src_key_padding_mask=(seq == 0))
        return self.ln(x)

    def forward(self, seq):
        h = self._encode(seq)
        interests = self._extract(h, (seq == 0), self.K)                 # (B, K, d)
        return self.out(interests).max(dim=1).values                     # max aggregation

    def final_logits(self, seq):
        return self.forward(seq)

    def forward_multi_pos(self, seq, pos):
        """pos: (B, S) -> (B*S, n_items). One encoding, S extractions."""
        h = self._encode(seq)
        B, _, d = h.shape
        I = self._interests_at(h, seq, pos)                              # (B, S, K, d)
        return self.out(I.reshape(-1, self.K, d)).max(dim=1).values

    def reg_loss(self, seq):
        interests = self._extract(self._encode(seq), (seq == 0), self.K)
        return self.div_weight * self.diversity_loss(interests)


class REMI(SeqRecModel, MultiInterestMixin):
    """REMI (lite reproduction): multi-interest extraction + interest-level attention.

    The original REMI adds relative-position and causal interest disentanglement;
    we keep the identifiable core (multiple interests, a diversity regularizer, and
    target-aware interest aggregation) as the representative "multi-interest but no
    test-time adaptation" baseline. Differences are stated in the paper appendix.
    """

    name = "REMI"
    train_on_final = True

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, n_heads=2,
                 dropout=0.2, K=4, div_weight=0.01, tau=0.1):
        super().__init__()
        self.emb = ItemEmbedding(n_items, d_model, max_len, dropout)
        self.K = K
        self.div_weight = div_weight
        self.tau = tau
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.interest_query = nn.Parameter(torch.randn(K, d_model) * 0.02)
        self.agg = nn.Linear(d_model, d_model, bias=False)
        self.out = _Scorer(self.emb.item)

    def _encode(self, seq):
        x = self.emb(seq)
        m = torch.triu(torch.ones(seq.size(1), seq.size(1), dtype=torch.bool,
                                  device=seq.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=m, src_key_padding_mask=(seq == 0))
        return self.ln(x)

    def _aggregate(self, h, interests, last):
        """Target-aware weighted average of the interest vectors."""
        w = torch.softmax(
            torch.einsum("bd,bkd->bk", self.agg(last), interests) / math.sqrt(h.size(-1)),
            dim=-1)
        return torch.einsum("bk,bkd->bd", w, interests)

    def forward(self, seq):
        h = self._encode(seq)
        B = h.size(0)
        idx = (seq != 0).sum(dim=1) - 1
        last = h[torch.arange(B, device=seq.device), idx.clamp(min=0)]
        interests = self._extract(h, (seq == 0), self.K)
        return self.out(self._aggregate(h, interests, last))

    def final_logits(self, seq):
        return self.forward(seq)

    def forward_multi_pos(self, seq, pos):
        """pos: (B, S) -> (B*S, n_items). One encoding, S extractions."""
        h = self._encode(seq)
        B, T, d = h.shape
        S = pos.size(1)
        I = self._interests_at(h, seq, pos)                              # (B, S, K, d)
        last = h[torch.arange(B, device=seq.device).unsqueeze(1),
                 pos.clamp(max=T - 1)]                                   # (B, S, d)
        return self.out(self._aggregate(h, I.reshape(B * S, self.K, d),
                                        last.reshape(B * S, d)))

    def reg_loss(self, seq):
        interests = self._extract(self._encode(seq), (seq == 0), self.K)
        return self.div_weight * self.diversity_loss(interests)
