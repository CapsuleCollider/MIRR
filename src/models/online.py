"""Test-time-training baselines: TTT4Rec / T^2ARec / DT3OR, and the shared online backbone.

Streaming protocol (identical for every online method).
    For a user sequence ``x_1..x_n`` the model reads from ``t = 1`` to ``t = n-1``.
    At every step it first emits the prediction for ``x_{t+1}``; when ``t+1 == n``
    that prediction is the leave-one-out test target and its metrics are recorded,
    otherwise the *already observed* ``x_{t+1}`` is used to build a self-supervised
    loss and take one test-time update. The held-out target ``x_n`` never enters any
    update.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ItemEmbedding, SeqRecModel
from .ssm import SSMBlock


class _Scorer(nn.Module):
    def __init__(self, item_emb: nn.Embedding):
        super().__init__()
        self.item_emb = item_emb
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        E = self.item_emb.weight
        if h.dim() == 3:
            return torch.einsum("bkd,nd->bkn", h, E) * self.scale
        return (h @ E.T) * self.scale


def inbatch_nce(u: torch.Tensor, pos_emb: torch.Tensor, tau: float = 0.07):
    """In-batch contrastive alignment between the user state and the observed next item."""
    u = F.normalize(u, dim=-1)
    p = F.normalize(pos_emb, dim=-1)
    logits = (u @ p.T) / tau
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)


def ce_loss(logits: torch.Tensor, y: torch.Tensor):
    return F.cross_entropy(logits, y)


# --------------------------------------------------------------------------- #
# Shared online backbone
# --------------------------------------------------------------------------- #
class OnlineBackbone(SeqRecModel):
    """K parallel SSM slots (shared parameters, independent states) + attention pooling.

    ``K = 1`` degenerates to a single-state SSM, which is the backbone of T^2ARec and
    TTT4Rec. Subclasses decide how slots are written by overriding
    :meth:`step_coeffs` (routing weights ``r`` and update gate ``g``) and how items
    are scored by overriding :meth:`score`.
    """

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, dropout=0.2,
                 K=1, d_state=None):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.max_len = max_len
        self.K = K
        self.emb = ItemEmbedding(n_items, d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [SSMBlock(d_model, d_state, dropout=dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.out = _Scorer(self.emb.item)
        self.pool_q = nn.Parameter(torch.randn(d_model) * 0.02)

    # ---------------- state ---------------- #
    def init_state(self, B, device):
        return [b.init_state(B, self.K, device) for b in self.blocks]

    def detach_state(self, state):
        """Truncated BPTT: detach every recurrent buffer from the current graph.

        Subclasses that maintain additional stepwise buffers (MIRR keeps the recent
        interaction window) must override this, otherwise the second ``backward()``
        raises "Trying to backward through the graph a second time".
        """
        return [s.detach() if torch.is_tensor(s) else s for s in state]

    # ---------------- pooling ---------------- #
    def _pool(self, H, return_w=False):
        """H: (B, K, d) -> u: (B, d)."""
        self._last_H = H
        self._last_u = H[:, 0] if self.K == 1 else torch.einsum(
            "bk,bkd->bd",
            torch.softmax(torch.einsum("bkd,d->bk", H, self.pool_q) / math.sqrt(H.size(-1)),
                          dim=-1), H)
        return (self._last_u, None) if return_w else self._last_u

    # ---------------- training-time parallel forward ---------------- #
    def forward(self, seq):
        B, T = seq.shape
        x = self.emb(seq)                       # (B, T, d)
        for b in self.blocks:
            x, _ = b(x)
        x = self.ln(x)
        u = x if self.K == 1 else self._pool(
            x.reshape(B * T, self.K, self.d_model)).reshape(B, T, self.d_model)
        return self.out(u)                      # (B, T, n_items)

    def final_logits(self, seq):
        logits = self.forward(seq)
        B = seq.size(0)
        idx = (seq != 0).sum(dim=1) - 1
        return logits[torch.arange(B, device=seq.device), idx.clamp(min=0)]

    # ---------------- one online step ---------------- #
    def step_coeffs(self, xt, H, pos):
        """Subclasses return (r, g): routing weights (B, K, 1) and update gate (B, K, 1).

        ``None`` means "no routing": the plain single-state recurrence.
        """
        return None, None

    def _embed_at(self, x_t, pos):
        xt = self.emb.item(x_t)
        if pos is not None:
            pe = self.emb.pos(torch.as_tensor(pos, device=xt.device))
            if pe.dim() == 1:                    # scalar position
                pe = pe.unsqueeze(0)
            xt = xt + pe
        return self.emb.drop(self.emb.norm(xt))                   # (B, d)

    def step(self, x_t, state, pos=None, **kw):
        """x_t: (B,) item ids; state: per-layer (B, K, D). -> (logits, new_state, info)."""
        return self._step_vec(self._embed_at(x_t, pos), state, pos, **kw)

    def _step_vec(self, xt, state, pos=None, **kw):
        """xt: (B, d) already-embedded current input."""
        B = xt.size(0)
        r, g = self.step_coeffs(xt, state[-1], pos)
        cur = xt.unsqueeze(1).expand(B, self.K, self.d_model)     # (B, K, d)
        new_state = []
        for i, b in enumerate(self.blocks):
            a, w, v = b.coeffs(b.norm1(cur))                      # (B, K, D)
            if r is not None:                          # r, g are both (B, n_slots, 1)
                alpha = (1.0 - g * r * (1.0 - a)).clamp(0.0, 1.0)
                beta = g * r * w
            else:
                alpha, beta = a, w
            y, h_new = b.apply(state[i], alpha, beta, v)
            o = cur + y
            cur = o + b.ffn(b.norm2(o))
            new_state.append(h_new)
        cur = self.ln(cur)                                        # (B, K, d)
        u = self._pool(cur)
        return self.score(cur, u), new_state, {"r": r, "g": g}

    # ---------------- scoring ---------------- #
    def score(self, H, u):
        """Items scores from the slot states H: (B, K, d) and pooled vector u: (B, d)."""
        return self.out(u)


# --------------------------------------------------------------------------- #
# TTT4Rec: single state + one reconstruction-style update at test time
# --------------------------------------------------------------------------- #
class TTT4Rec(OnlineBackbone):
    name = "TTT4Rec"
    is_ttt = True

    def __init__(self, n_items, **kw):
        kw["K"] = 1
        super().__init__(n_items, **kw)

    def ttt_param_groups(self):
        emb_ids = {id(p) for p in self.emb.item.parameters()}
        return [p for p in self.parameters() if id(p) not in emb_ids]

    def ttt_closure(self, logits, y, info=None, model=None):
        return ce_loss(logits, y)


# --------------------------------------------------------------------------- #
# T^2ARec: single-state SSM + two alignment objectives
# --------------------------------------------------------------------------- #
class T2ARec(OnlineBackbone):
    name = "T^2ARec"
    is_ttt = True

    def __init__(self, n_items, align_w=0.1, tau=0.07, **kw):
        kw["K"] = 1
        super().__init__(n_items, **kw)
        self.align_w = align_w
        self.tau = tau

    def ttt_param_groups(self):
        emb_ids = {id(p) for p in self.emb.item.parameters()}
        return [p for p in self.parameters() if id(p) not in emb_ids]

    def ttt_closure(self, logits, y, info=None, model=None):
        l = ce_loss(logits, y)
        l = l + self.align_w * inbatch_nce(model._last_u, model.emb.item(y), self.tau)
        return l


# --------------------------------------------------------------------------- #
# DT3OR: K cluster slots + frozen item-level cluster router + dual test-time tasks
# --------------------------------------------------------------------------- #
class DT3OR(OnlineBackbone):
    name = "DT3OR"
    is_ttt = True

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, dropout=0.2,
                 K=4, d_state=None, align_w=0.1, sd_w=0.1, tau=0.07):
        super().__init__(n_items, d_model=d_model, max_len=max_len, n_layers=n_layers,
                         dropout=dropout, K=K, d_state=d_state)
        # Item-level cluster router: depends only on x_t and is frozen after training,
        # which is exactly the "quasi-static multi-cluster" write-time behaviour.
        self.cluster_router = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, K))
        # The only module updated at test time.
        self.adapter = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.align_w, self.sd_w, self.tau = align_w, sd_w, tau

    def _cluster(self, xt):
        return torch.softmax(self.cluster_router(xt), dim=-1)     # (..., K)

    def _pool(self, H, return_w=False):
        u, w = super()._pool(H, return_w=True)
        u = self.adapter(u)
        self._last_u = u
        return (u, w) if return_w else u

    def step_coeffs(self, xt, H, pos):
        B = xt.size(0)
        r = self._cluster(xt).unsqueeze(-1)                       # (B, K, 1)
        g = torch.ones(B, self.K, 1, device=xt.device)            # no update gate
        return r, g

    def forward(self, seq):
        """Training: closed-form parallel scan under the static item-level routing."""
        B, T = seq.shape
        x = self.emb(seq)                                          # (B, T, d)
        r = self._cluster(x.reshape(B * T, -1)).reshape(B, T, self.K)
        rr = r.unsqueeze(-1)                                       # (B, T, K, 1)
        cur = x.unsqueeze(2)                                       # (B, T, 1, d)
        for b in self.blocks:
            a, w, v = b.coeffs(b.norm1(cur))                       # (B, T, [1|K], D)
            al = 1.0 - rr * (1.0 - a)                              # (B, T, K, D)
            be = rr * w
            log_a = torch.log(al.clamp(min=1e-4))
            logP = torch.cumsum(log_a, dim=1).clamp(min=-13.8)
            P = torch.exp(logP)
            S = torch.cumsum(be * torch.exp(-logP), dim=1)
            h = P * S                                              # (B, T, K, D)
            y = b.out_proj(h * F.silu(v))                          # (B, T, K, d)
            cur = cur + y
            cur = cur + b.ffn(b.norm2(cur))
        cur = self.ln(cur)                                         # (B, T, K, d)
        u = self._pool(cur.reshape(B * T, self.K, self.d_model)).reshape(B, T, self.d_model)
        return self.out(u)

    def final_logits(self, seq):
        logits = self.forward(seq)
        B = seq.size(0)
        idx = (seq != 0).sum(dim=1) - 1
        return logits[torch.arange(B, device=seq.device), idx.clamp(min=0)]

    def ttt_param_groups(self):
        return list(self.adapter.parameters())

    def ttt_closure(self, logits, y, info=None, model=None):
        l = ce_loss(logits, y)
        l = l + self.align_w * inbatch_nce(model._last_u, model.emb.item(y), self.tau)
        # Self-distillation: the strongest cluster (teacher, detached) -> cluster mean (student).
        H = model._last_H                                          # (B, K, d)
        slot_logits = self.out(H)                                  # (B, K, n_items)
        with torch.no_grad():
            p_t = F.log_softmax(slot_logits.max(dim=1).values.detach(), dim=-1)
        p_s = F.log_softmax(slot_logits.mean(dim=1), dim=-1)
        return l + self.sd_w * F.kl_div(p_s, p_t, reduction="batchmean", log_target=True)
