"""MIRR: Multi-Interest Read-Time Router.

MIRR defers interest routing from *write time* to *read time*. It has two stages:

Stage 1 (long term).
    A single-state gated SSM compresses the whole history into one state, produces
    the long-term representation ``u_t``, and scores the full catalog. The top-M
    items become the candidate set ``C_t``.

Stage 2 (recent + multi-interest).
    The last ``m`` interactions are **kept outside the state** and re-read only when
    a prediction is requested. ``R`` learnable interest probes, conditioned on
    ``u_t``, attend over this window and produce one interest vector ``U_r`` per
    probe. Each interest scores every candidate, and a candidate keeps the score of
    the interest that fits it best, shifted by how much that interest agrees with
    ``u_t``.

Fusion.
    ``score(i) = (1 - a) * base(i) + a * l(i)`` for ``i`` in ``C_t``, and
    ``base(i)`` otherwise, so the final ranking still covers the full catalog.

Nothing derived from Stage 2 is written back into the state, and no parameter is
updated at serving time: a new interaction advances the state by one step and shifts
the window by one position.

Two ablation regimes are supported by the same class:

* ``--read_win 0`` turns Stage 2 off. With ``--K 1`` this is the Stage-1-only
  model; with ``--K 4`` it is **MIRR-W**, the write-time router that assigns each
  interaction to one of K recurrent slots as it is written. MIRR-W is the
  write-time对照 used throughout the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online import OnlineBackbone


class MIRR(OnlineBackbone):
    name = "MIRR"
    # MIRR adapts by changing what it reads, not by updating parameters.
    is_ttt = False

    def __init__(self, n_items, d_model=64, max_len=50, n_layers=2, dropout=0.2,
                 K=1, read_win=8, read_probes=8, read_topm=-1,
                 read_cond=True, per_user_alpha=False, route_bias=True, bptt=0):
        """Args:
            n_items: catalog size (index 0 is padding).
            K: number of recurrent slots. Forced to 1 when ``read_win > 0``, because
                read-time routing keeps the long-term state free of interest structure.
            read_win: ``m``, number of recent interactions kept outside the state and
                re-read at prediction time. ``0`` disables Stage 2.
            read_probes: ``R``, number of interest probes (an upper bound on the number
                of interests separable at one prediction).
            read_topm: ``M``, candidate-list size. ``-1`` selects it from the catalog
                with the paper's rule ``M = max(512, ceil(0.04 * |I|))``.
            read_cond: condition the probes on the long-term state (Eq. 2 in the paper).
                Turning it off makes the probes a fixed global weighted average of the
                window, which is not routing.
            per_user_alpha: make the fusion weight sample-dependent,
                ``a_t = sigmoid(eta + MLP([u_t; mean_r U_r]))`` with a zero-initialised
                last layer. The two variants are selected on validation data.
            route_bias: let ``u_t`` bias which interest a candidate is assigned to
                (Eq. 6 in the paper).
            bptt: truncate backpropagation through time every ``bptt`` steps (0 = never).
        """
        self.read_win = int(read_win)
        self.n_probe = int(read_probes) if self.read_win > 0 else 0
        self.read_cond = bool(read_cond)
        # 0 = global gate, 1 = naive per-user shift, 2 = batch-standardised per-user
        self.per_user_alpha = int(per_user_alpha)
        self.route_bias = bool(route_bias) and self.read_win > 0
        self.bptt = int(bptt)
        # A read-time router keeps a single long-term state; slots are the write-time
        # ablation only.
        if self.read_win > 0:
            K = 1
        self.n_int = K

        super().__init__(n_items, d_model=d_model, max_len=max_len, n_layers=n_layers,
                         dropout=dropout, K=K)

        # Candidate-list size (Section 4.5): long enough to contain the target.
        if int(read_topm) < 0:
            cand = max(512, math.ceil(0.04 * n_items))
        else:
            cand = int(read_topm)
        self.cand_size = int(min(max(cand, 1), n_items))

        # ---- write-time routing (only used when K > 1) ----
        if self.n_int > 1:
            d_state = self.blocks[0].d_state
            self.router = nn.Sequential(
                nn.Linear(d_model + self.n_int * d_state, d_model), nn.Tanh(),
                nn.Linear(d_model, self.n_int))
            self.pool_q_proj = nn.Linear(d_model, d_model, bias=False)
        else:
            self.router = None
            self.pool_q_proj = None

        # ---- read-time routing (Stage 2) ----
        self._win = None            # (B, m, d) recent interactions, reset per batch
        self._win_mask = None       # (B, m) bool, which window entries are real
        if self.read_win > 0:
            self.probes = nn.Parameter(torch.randn(self.n_probe, d_model) * 0.02)
            # W_q: maps u_t to a per-probe adjustment (Eq. 2).
            self.probe_cond = nn.Linear(d_model, self.n_probe * d_model)
            # eta: logit of the fusion weight a = sigmoid(eta), eta_0 = -2 -> a_0 ~ 0.12,
            # so training starts close to the Stage-1-only model.
            self.logit_alpha = nn.Parameter(torch.tensor(-2.0))
            if self.per_user_alpha:
                self.alpha_mlp = nn.Sequential(
                    nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
                nn.init.zeros_(self.alpha_mlp[-1].weight)
                nn.init.zeros_(self.alpha_mlp[-1].bias)
            if self.per_user_alpha >= 2:
                # beta: strength of the standardised per-user re-allocation.
                self.alpha_beta = nn.Parameter(torch.tensor(0.3))
            if self.route_bias:
                # W_b: maps u_t into the interest space (Eq. 5).
                self.interest_bias = nn.Linear(d_model, d_model)
                # gamma: log-parameter of the bias strength, lambda = softplus(gamma),
                # gamma_0 = -2 -> lambda_0 ~ 0.13.
                self.log_lambda = nn.Parameter(torch.tensor(-2.0))

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def init_state(self, B, device):
        self._win = None                       # reallocated lazily on the first step
        self._win_mask = None
        return [b.init_state(B, self.K, device) for b in self.blocks]

    def detach_state(self, state):
        """Truncated BPTT. The recent window is a stepwise buffer too, so it must be
        detached as well; missing it triggers "Trying to backward through the graph a
        second time" during streaming evaluation."""
        if self._win is not None:
            self._win = self._win.detach()
        return [s.detach() if torch.is_tensor(s) else s for s in state]

    def _push_win(self, xt, real):
        """Shift the recent-interaction window by one position. xt: (B, d)."""
        if self.read_win <= 0:
            return
        B, d = xt.size(0), xt.size(-1)
        if self._win is None or self._win.shape[0] != B:
            self._win = xt.new_zeros(B, self.read_win, d)
            self._win_mask = xt.new_zeros(B, self.read_win, dtype=torch.bool)
        self._win = torch.cat([self._win[:, 1:], xt.unsqueeze(1)], dim=1)
        self._win_mask = torch.cat([self._win_mask[:, 1:], real.unsqueeze(1)], dim=1)

    # ------------------------------------------------------------------ #
    # Routing / gating
    # ------------------------------------------------------------------ #
    def step_coeffs(self, xt, H, pos):
        """Write-time routing weights r (B, K, 1). MIRR has no update gate."""
        self._cur_xt = xt
        if self.n_int <= 1:
            return None, None
        B, K = xt.size(0), self.n_int
        Hin = H.reshape(B, K * H.size(-1))
        r = torch.softmax(self.router(torch.cat([xt, Hin], dim=-1)), dim=-1).unsqueeze(-1)
        g = torch.ones(B, K, 1, device=xt.device)
        return r, g

    def _pool(self, H, return_w=False):
        """Target-aware pooling: the query is produced from the current interaction, so
        the model picks the slot that is active *now* rather than an input-independent
        average."""
        self._last_H = H
        if self.n_int == 1:
            self._last_u = H[:, 0]
            return (self._last_u, None) if return_w else self._last_u
        xt = getattr(self, "_cur_xt", None)
        q = self.pool_q_proj(xt) if xt is not None else self.pool_q
        w = torch.softmax(torch.einsum("bkd,bd->bk", H, q) / math.sqrt(H.size(-1)), dim=-1)
        self._last_u = torch.einsum("bk,bkd->bd", w, H)
        return (self._last_u, w) if return_w else self._last_u

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #
    def _probes(self, u):
        """Q_r = q_r + (W_q u)_r  ->  (B, R, d). Eq. 2."""
        B = self._win.size(0)
        if not self.read_cond or u is None:
            return self.probes.unsqueeze(0).expand(B, -1, -1)
        return self.probes.unsqueeze(0) + self.probe_cond(u).view(B, self.n_probe, -1)

    def _interest_vectors(self, u=None):
        """U_r = sum_j A_{r,j} x_j  ->  (B, R, d). Eq. 3."""
        B, m, d = self._win.shape
        Q = self._probes(u)                                            # (B, R, d)
        s = torch.einsum("brd,bmd->brm", Q, self._win) / math.sqrt(d)  # (B, R, m)
        mask = self._win_mask
        # A window that is entirely padding would produce a row of -inf; unmask one
        # entry so the softmax stays finite.
        mask = mask | (~mask.any(dim=1, keepdim=True))
        s = s.masked_fill(~mask.unsqueeze(1), float("-inf"))
        a = torch.softmax(s, dim=-1)
        return torch.einsum("brm,bmd->brd", a, self._win)

    def _route_max(self, L_all, U, u):
        """l_t(i) = max_r ( l_r(i) + lambda * b_r ).  L_all: (B, R, M). Eq. 6."""
        if not self.route_bias:
            return L_all.max(dim=1).values
        q = self.interest_bias(u)                                      # W_b u
        b = torch.einsum("brd,bd->br", U, q) / math.sqrt(U.size(-1))  # b_r
        lam = F.softplus(self.log_lambda)
        return (L_all + lam * b.unsqueeze(-1)).max(dim=1).values

    def _alpha(self, u, U):
        """Fusion weight a: scalar, or (B, 1) in the per-user variants. Eq. 7-8.

        per_user_alpha:
          0 -> one global gate, sigmoid(logit_alpha)
          1 -> naive per-user shift, sigmoid(logit_alpha + f([u; ubar]))
          2 -> batch-standardised re-allocation,
               sigmoid(logit_alpha + beta * (f - mean f) / std f)

        Why 2 exists: variant 1 lets a small MLP shift the BATCH MEAN of the fusion
        weight, which short-circuits the global logit_alpha that training already
        tuned.  On datasets where the global gate is already right, that is pure
        added variance.  Centring removes the batch-level shift, so 2 only
        re-allocates the weight ACROSS users and never moves its mean.  Its
        hypothesis space therefore STRICTLY CONTAINS the global gate (beta -> 0
        recovers it exactly), i.e. the worst case of 2 is 0, not worse than 0.

        std is detached: with a zero-initialised last layer raw is identically 0 at
        the start of training, so std(raw) = 0 and d std/dx = (x - mean)/(n*std)
        = 0/0 = NaN would poison the whole run.  std is only a rescaling constant
        and needs no gradient.  unbiased=False as well, because a batch of size 1
        has 0 degrees of freedom and returns NaN.
        """
        if not self.per_user_alpha:
            return torch.sigmoid(self.logit_alpha)
        ubar = U.mean(dim=1)                                           # (B, d)
        raw = self.alpha_mlp(torch.cat([u, ubar], dim=-1))             # (B, 1)
        if self.per_user_alpha >= 2:
            sd = torch.nan_to_num(raw.std(unbiased=False).detach(),
                                  nan=1.0, posinf=1.0).clamp(min=1e-4)
            raw = (raw - raw.mean().detach()) / sd * self.alpha_beta
        return torch.sigmoid(self.logit_alpha + raw)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_base(self, H, u):
        """Stage 1: score the whole catalog from the long-term representation."""
        return self.out(u)

    def score(self, H, u):
        """Stage 1 + Stage 2 fused. Returns full-catalog logits (B, n_items). Eq. 7."""
        base = self._score_base(H, u)
        if self.read_win <= 0 or self._win is None:
            return base
        N = base.size(-1)
        M = min(self.cand_size, N)
        topv, topi = torch.topk(base, M, dim=-1)                       # (B, M)
        Esub = self.out.item_emb.weight[topi]                          # (B, M, d)
        U = self._interest_vectors(u)                                  # (B, R, d)
        L_all = torch.einsum("brd,bmd->brm", U, Esub) * self.out.scale  # l_r(i)
        ell = self._route_max(L_all, U, u)                             # (B, M)
        a = self._alpha(u, U)
        # Convex combination: keeps both scores on one scale and makes d score / d a
        # equal to (l - base), so a directly reflects how the two stages disagree.
        newv = topv + a * (ell - topv)
        return base.scatter(-1, topi, newv)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, seq):
        """Stepwise forward over the sequence.

        Returns:
            ``(loss_sum, n_targets, logits_last)`` where the loss is the next-item
            cross-entropy accumulated over every supervised position of the batch and
            ``logits_last`` is the prediction made at each sequence's last valid step.
        """
        B, T = seq.shape
        x = self.emb(seq)                                       # (B, T, d)
        state = self.init_state(B, seq.device)
        T_eff = max(int((seq != 0).sum(dim=1).max().item()), 1)
        last_idx = (seq != 0).sum(dim=1).clamp(min=1) - 1
        loss_sum = seq.new_zeros((), dtype=torch.float32)
        n_tgt = 0
        logits_last = torch.zeros(B, self.n_items, device=seq.device)
        for t in range(T_eff):
            self._push_win(x[:, t, :], seq[:, t] != 0)
            logits, state, _ = self._step_vec(
                x[:, t, :], state,
                pos=torch.full((B,), min(t, self.max_len - 1), device=seq.device))
            done = (last_idx == t)
            if done.any():
                logits_last[done] = logits[done]
            if t + 1 < T_eff:
                tgt = seq[:, t + 1]
                m = tgt != 0
                if m.any():
                    loss_sum = loss_sum + F.cross_entropy(logits[m], tgt[m],
                                                          reduction="sum")
                    n_tgt += int(m.sum())
            if self.bptt and (t + 1) % self.bptt == 0:
                state = self.detach_state(state)                # truncated BPTT
        return loss_sum, n_tgt, logits_last

    def final_logits(self, seq):
        """Inference path: score the catalog ONCE, at the last step of each sequence.

        ``forward()`` has to produce (B, n_items) logits at every position because
        supervision is per-position, but inference keeps only the last one, so T-1
        scorings are computed and thrown away.  Here we advance the state with
        ``_encode_step`` and defer ``score()`` to the step at which each sequence
        actually ends: B scorings instead of B*T.  Bit-identical to
        ``forward()[2]``, it just stops doing discarded work.
        """
        B, T = seq.shape
        x = self.emb(seq)
        state = self.init_state(B, seq.device)
        T_eff = max(int((seq != 0).sum(dim=1).max().item()), 1)
        last_idx = (seq != 0).sum(dim=1).clamp(min=1) - 1
        logits_last = torch.zeros(B, self.n_items, device=seq.device)
        for t in range(T_eff):
            self._push_win(x[:, t, :], seq[:, t] != 0)
            cur, u, state, _ = self._encode_step(
                x[:, t, :], state,
                pos=torch.full((B,), min(t, self.max_len - 1), device=seq.device))
            done = (last_idx == t)
            if done.any():
                # Score only the sequences that just ended.  The read-time branch
                # derives several shapes from self._win's batch dim, so the window
                # must be sliced along with it (absent when read_win == 0).
                win, wmask = self._win, self._win_mask
                if win is not None:
                    self._win, self._win_mask = win[done], wmask[done]
                logits_last[done] = self.score(cur[done], u[done])
                if win is not None:
                    self._win, self._win_mask = win, wmask
        return logits_last

    def step(self, x_t, state, pos=None, **kw):
        """One streaming step: advance the recent window, then the recurrent state."""
        if self.read_win > 0:
            xt = self._embed_at(x_t, pos)
            self._push_win(xt, x_t != 0)
            return self._step_vec(xt, state, pos, **kw)
        return super().step(x_t, state, pos, **kw)
