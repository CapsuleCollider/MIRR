"""Slot-collapse diagnostics (appendix).

MIRR's motivation is that write-time routing collapses: when K recurrent slots share
their recurrence parameters, the routing distribution stays uniform, the slots receive
symmetric gradients, and they end up as copies of one another. This module measures the
same quantities on *external* multi-interest models (ComiRec / REMI / DT3OR) and on the
write-time variant MIRR-W, so the finding cannot be dismissed as an artefact of our own
implementation.

Two families of metrics are reported:

* **Normalised redundancy.** ``cos_within`` (mean pairwise cosine of one user's slots)
  is meaningless on its own, because representations are anisotropic and even two random
  users look similar. We therefore subtract a cross-user baseline ``cos_cross`` and
  normalise: ``R_norm = (cos_within - cos_cross) / (1 - cos_cross)``, which is 0 when a
  user's slots are no more alike than two random users' and 1 when they are copies.
  ``n_eff = exp(H(svd spectrum))`` is the R-independent companion: 1 means one effective
  interest, K means fully separated.
* **Functional redundancy.** ``jaccard_top50`` measures how much the top-50 retrieved by
  two slots overlap; ``load_share_max`` and ``n_slots_used`` measure whether the routing
  actually spreads items across slots. The **mean-substitution test** replaces all slots
  by their mean and re-measures: if accuracy does not move, the slots were redundant.

Note on cross-model comparability: ComiRec/REMI interest vectors come from a
Transformer output while SSM slots come from a recurrent state, so absolute values are
not comparable across models. Only the normalised and functional metrics are.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


def _upper_pairs(K, device):
    return torch.triu_indices(K, K, offset=1, device=device)


def _chunks(n, c):
    for i in range(0, n, c):
        yield i, min(i + c, n)


def ms(x):
    """Mean, sample standard deviation, count."""
    v = x.detach().reshape(-1).float().cpu().tolist() if torch.is_tensor(x) \
        else [float(t) for t in x]
    v = [t for t in v if t == t]
    if not v:
        return float("nan"), float("nan"), 0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / max(1, len(v) - 1))
    return m, sd, len(v)


def stat(x):
    """-> {"mean", "std", "n"}"""
    m, sd, n = ms(x)
    return {"mean": m, "std": sd, "n": n}


# --------------------------------------------------------------------------- #
# Representation-level redundancy
# --------------------------------------------------------------------------- #
def cos_within(I: torch.Tensor):
    """Mean pairwise cosine among one user's slots -> (B,). NaN when K < 2."""
    if I.size(1) < 2:
        return I.new_full((I.size(0),), float("nan")).float()
    Z = F.normalize(I.float(), dim=-1)
    S = torch.einsum("bkd,bld->bkl", Z, Z)
    i, j = _upper_pairs(I.size(1), I.device)
    return S[:, i, j].mean(dim=1)


def cos_cross(I: torch.Tensor, n_pairs: int = 8192, seed: int = 0):
    """Anisotropy baseline: cosine between slots of *different* users.

    Returns ``(same, rnd)``:
      * ``same`` — slot k of user a vs. slot k of user b. The slot query is shared
        across users, so same-indexed slots are naturally more alike; this baseline is
        conservative and yields a smaller R_norm (harder to call collapse).
      * ``rnd``  — both the users and the slot indices are drawn independently, a more
        neutral baseline.

    Both are reported, because the choice of denominator is a legitimate question.
    """
    B, K, _ = I.shape
    dev = I.device
    Z = F.normalize(I.float(), dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = max(64, min(n_pairs, B * 8))
    a = torch.randint(0, B, (n,), generator=g)
    b = (a + 1 + torch.randint(0, max(B - 1, 1), (n,), generator=g)) % B
    ka = torch.randint(0, K, (n,), generator=g)
    kb = torch.randint(0, K, (n,), generator=g)
    a, b, ka, kb = a.to(dev), b.to(dev), ka.to(dev), kb.to(dev)
    return float((Z[a, ka] * Z[b, ka]).sum(dim=-1).mean()), \
           float((Z[a, ka] * Z[b, kb]).sum(dim=-1).mean())


def r_norm(within: float, cross: float):
    """Normalised redundancy: 0 = no more alike than random users, 1 = exact copies."""
    return (within - cross) / max(1.0 - cross, 1e-6)


def n_eff(I: torch.Tensor):
    """Effective number of interests = exp(entropy of the squared singular values).

    Unlike the pairwise metrics this is defined for every K (including K = 1) and is
    comparable across different K, so it is the metric to use when sweeping R.
    """
    s = torch.linalg.svdvals(I.float())                    # (B, min(K, d))
    p = s ** 2
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return torch.exp(-(p * (p + 1e-12).log()).sum(dim=-1))


# --------------------------------------------------------------------------- #
# Functional redundancy
# --------------------------------------------------------------------------- #
def chunked_topk(score_fn: Callable, n_items: int, K: int, B: int, k: int,
                 device, chunk_items: int = 8192, chunk_users: int = 512):
    """Full-catalog top-k per (user, slot) -> (B, K, k). Chunked to bound memory."""
    out = torch.zeros(B, K, k, dtype=torch.long, device=device)
    for u0, u1 in _chunks(B, chunk_users):
        v = ix = None
        for c0, c1 in _chunks(n_items, chunk_items):
            s = score_fn(slice(u0, u1), c0, c1)
            tv, ti = torch.topk(s, k=min(k, s.size(-1)), dim=-1)
            v, ix = (tv, ti + c0) if v is None else (torch.cat([v, tv], dim=-1),
                                                     torch.cat([ix, ti + c0], dim=-1))
            v, order = torch.topk(v, k=min(k, v.size(-1)), dim=-1)
            ix = ix.gather(-1, order)
        out[u0:u1] = ix
    return out


def jaccard_topk(topk_idx: torch.Tensor, chunk_users: int = 512):
    """Mean pairwise Jaccard of the per-slot top-k lists -> (B,). NaN when K < 2."""
    B, K, k = topk_idx.shape
    if K < 2:
        return torch.full((B,), float("nan"))
    i, j = _upper_pairs(K, topk_idx.device)
    res = []
    for u0, u1 in _chunks(B, chunk_users):
        t = topk_idx[u0:u1]
        acc = []
        for a, b in zip(i.tolist(), j.tolist()):
            A = t[:, a, :].unsqueeze(2)                       # (b, k, 1)
            Bm = t[:, b, :].unsqueeze(1)                      # (b, 1, k)
            inter = (A == Bm).any(dim=2).sum(dim=1).float()
            acc.append(inter / (2.0 * k - inter).clamp(min=1e-6))
        res.append(torch.stack(acc, dim=1).mean(dim=1))
    return torch.cat(res, dim=0)


def load_share_stats(share: torch.Tensor):
    """share: (K,) fraction of items won by each slot -> (max share, entropy / ln K)."""
    K = share.size(0)
    mx = float(share.max())
    ent = 0.0 if K < 2 else float(-(share * (share + 1e-12).log()).sum() / math.log(K))
    return mx, ent


def _rank(x: torch.Tensor):
    N = x.size(-1)
    order = torch.argsort(x, dim=-1)
    r = torch.empty_like(x)
    r.scatter_(-1, order, torch.arange(N, dtype=x.dtype, device=x.device).expand(x.size(0), N))
    return r + 1.0


def _pearson_row(a, b):
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-12)


def spearman_scores(lo: torch.Tensor, lc: torch.Tensor, chunk_users: int = 512):
    """Per-user Spearman correlation of the full score vector before/after substitution.

    Recall@10 is a step function: it stays at 0.000 as long as each user's top-10 set is
    unchanged, even if the ranking was reshuffled underneath. A Spearman of 1.0 shows
    that an observed delta of 0 is *real* (the substitution changed the ranking by
    nothing) rather than an artefact (the substitution never took effect).
    """
    out = []
    for u0, u1 in _chunks(lo.size(0), chunk_users):
        a, b = lo[u0:u1].float(), lc[u0:u1].float()
        out.append(_pearson_row(_rank(a), _rank(b)))
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
# Slot extraction (read-only: no forward logic is modified)
# --------------------------------------------------------------------------- #
def model_kind(model):
    """-> 'comirec' | 'remi' | 'ssm' | 'read'"""
    n = model.__class__.__name__
    if n == "ComiRec":
        return "comirec"
    if n == "REMI":
        return "remi"
    if getattr(model, "read_win", 0) > 0:
        return "read"
    return "ssm"


def slot_score_fn(model, I):
    """Scoring function that reuses the model's own output layer (incl. temperature)."""
    E = model.out.item_emb.weight
    scale = model.out.scale

    def fn(uslice, c0, c1):
        return torch.einsum("bkd,nd->bkn", I[uslice], E[c0:c1]) * scale

    return fn


@torch.no_grad()
def extract_slots(model, seq, device):
    """Run the model over ``seq`` and return its slots / interest vectors.

    Returns dict with ``I`` (B, K, d), ``A`` (B, T, K) soft assignment per history
    position, ``Amask`` (B, T), and ``W`` (B, K) pooling weights.
    """
    kind = model_kind(model)
    model.eval()
    B, T = seq.shape
    dev = seq.device

    if kind in ("comirec", "remi"):
        h = model._encode(seq)
        I = model._extract(h, (seq == 0), model.K)
        attn = model.last_attn                              # (B, K, T)
        A = attn.transpose(1, 2).clone()
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        out = dict(I=I.detach(), A=A.detach(), Amask=(seq != 0), W=None)
        if kind == "remi":
            idx = (seq != 0).sum(dim=1).clamp(min=1) - 1
            last = h[torch.arange(B, device=dev), idx]
            out["W"] = torch.softmax(
                torch.einsum("bd,bkd->bk", model.agg(last), I) / math.sqrt(h.size(-1)),
                dim=-1).detach()
        return out

    last_idx = (seq != 0).sum(dim=1).clamp(min=1) - 1
    T_eff = int(last_idx.max().item()) + 1
    n_slot = int(model.n_probe) if kind == "read" else int(model.K)
    d = int(model.d_model)
    I = torch.zeros(B, n_slot, d, device=dev)
    W = torch.zeros(B, n_slot, device=dev)
    rs = []
    if kind == "read":
        A = torch.zeros(B, model.read_win, n_slot, device=dev)
        Amask = torch.zeros(B, model.read_win, dtype=torch.bool, device=dev)
    state = model.init_state(B, dev)
    for t in range(T_eff):
        _, state, info = model.step(seq[:, t], state, pos=t)
        rs.append(info["r"])
        done = (last_idx == t)
        if not bool(done.any()):
            continue
        if kind == "read":
            u = model._last_u
            I[done] = model._interest_vectors(u)[done].detach()
            # Recompute the read-time attention without touching any state.
            Q = model._probes(u)
            s = torch.einsum("brd,bmd->brm", Q, model._win) / math.sqrt(d)
            mk = model._win_mask
            mk = mk | (~mk.any(dim=1, keepdim=True))
            s = s.masked_fill(~mk.unsqueeze(1), float("-inf"))
            A[done] = torch.softmax(s, dim=-1).transpose(1, 2)[done].detach()
            Amask[done] = model._win_mask[done].detach()
            W[done] = 1.0 / n_slot
        else:
            H = model._last_H
            I[done] = H[done].detach()
            _, w = model._pool(H, return_w=True)
            if w is not None:
                W[done] = w[done].detach()
    if kind == "read":
        return dict(I=I, A=A, Amask=Amask, W=W)
    K = int(model.K)
    A = torch.stack([r.detach().squeeze(-1) for r in rs], dim=1) if rs and rs[0] is not None \
        else torch.full((B, T_eff, K), 1.0 / K, device=dev)
    Tpad = max(T - A.size(1), 0)
    if Tpad:
        A = torch.cat([A, torch.full((B, Tpad, K), 1.0 / K, device=dev)], dim=1)
    Amask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    Amask[:, :T_eff] = True
    return dict(I=I, A=A, Amask=Amask & (seq != 0), W=W)


class Collapsed:
    """Context manager that replaces every slot by the mean of the slots.

    Which method is patched depends on where the slots are produced:
    ComiRec/REMI -> ``_extract``, MIRR (read-time) -> ``_interest_vectors``,
    SSM slots -> ``_pool`` (the mean is computed before pooling, then the original
    pooling and scoring run unchanged).
    """

    def __init__(self, model):
        self.model = model
        self.kind = model_kind(model)

    def __enter__(self):
        m = self.model
        if self.kind in ("comirec", "remi"):
            self.orig = m._extract

            def f(hseq, pad_mask, K, _o=self.orig):
                I = _o(hseq, pad_mask, K)
                return I.mean(dim=1, keepdim=True).expand_as(I)

            m._extract = f
        elif self.kind == "read":
            self.orig = m._interest_vectors

            def f(u=None, _o=self.orig):
                U = _o(u)
                return U.mean(dim=1, keepdim=True).expand_as(U)

            m._interest_vectors = f
        else:
            self.orig = m._pool

            def f(H, return_w=False, _o=self.orig):
                return _o(H.mean(dim=1, keepdim=True).expand_as(H), return_w=return_w)

            m._pool = f
        return self.model

    def __exit__(self, *a):
        key = {"comirec": "_extract", "remi": "_extract",
               "read": "_interest_vectors"}.get(self.kind, "_pool")
        setattr(self.model, key, self.orig)
        return False


@torch.no_grad()
def diagnose(model, loader, device, n_items, topk=50, max_users=None):
    """Run every diagnostic on ``loader`` and return a JSON-serialisable dict."""
    model.eval()
    acc = {k: [] for k in ("cos_within", "n_eff", "jacc", "spear")}
    Is, winners = [], []
    hit_o = {k: 0.0 for k in ("r10", "n10", "r20", "n20")}
    hit_c = {k: 0.0 for k in ("r10", "n10", "r20", "n20")}
    ntot = 0
    n_slots = None
    for batch in loader:
        seq, tgt = batch[0].to(device), batch[1].to(device)
        B = seq.size(0)
        if max_users is not None and ntot >= max_users:
            break
        b = extract_slots(model, seq, device)
        I = b["I"]
        n_slots = I.size(1)
        Is.append(I.detach().cpu())
        acc["cos_within"].append(cos_within(I).cpu())
        acc["n_eff"].append(n_eff(I).cpu())
        fn = slot_score_fn(model, I)
        acc["jacc"].append(jaccard_topk(chunked_topk(fn, n_items, n_slots, B, topk,
                                                     device)).cpu())
        # Winner: the slot that gives the target its highest score.
        E = model.out.item_emb.weight
        st = torch.einsum("bkd,bd->bk", I, E[tgt]) * model.out.scale
        winners.append(torch.nn.functional.one_hot(st.argmax(dim=-1), n_slots)
                       .float().cpu())
        # Mean-substitution test.
        lo = model.final_logits(seq)
        with Collapsed(model):
            lc = model.final_logits(seq)
        acc["spear"].append(spearman_scores(lo, lc).cpu())
        for lg, h in ((lo, hit_o), (lc, hit_c)):
            ts = lg.gather(1, tgt.unsqueeze(1))
            r = (lg > ts).sum(dim=1).float().cpu()
            for k, key in ((10, "r10"), (20, "r20")):
                h[key] += float((r < k).sum())
            h["n10"] += float((1.0 / torch.log2(r + 2.0) * (r < 10)).sum())
            h["n20"] += float((1.0 / torch.log2(r + 2.0) * (r < 20)).sum())
        ntot += B

    Icat = torch.cat(Is, dim=0).to(device)
    cross_same, cross_rand = cos_cross(Icat)
    cw = float(torch.cat(acc["cos_within"]).mean())
    share = torch.cat(winners, dim=0).mean(dim=0)
    share = share / share.sum().clamp(min=1e-12)
    mx, ent = load_share_stats(share)
    out = {
        "K": n_slots,
        "n_users": int(Icat.size(0)),
        "cos_within": round(cw, 5),
        "cos_cross_same": round(cross_same, 5),
        "cos_cross_rand": round(cross_rand, 5),
        "R_norm": round(r_norm(cw, cross_same), 4),
        "R_norm_rand": round(r_norm(cw, cross_rand), 4),
        "n_eff": stat(torch.cat(acc["n_eff"])),
        "jaccard_top50": stat(torch.cat(acc["jacc"])),
        "load_share_max": round(mx, 4),
        "load_share_entropy_ratio": round(ent, 4),
        "load_share_vec": [round(float(x), 4) for x in share],
        # slots that win at least half of a fair share are the ones actually in use
        "n_slots_used": int((share >= 1.0 / (2.0 * n_slots)).sum()),
        "metrics_orig": {k: v / max(ntot, 1) for k, v in hit_o.items()},
        "metrics_collapsed": {k: v / max(ntot, 1) for k, v in hit_c.items()},
        "delta": {k: (hit_c[k] - hit_o[k]) / max(ntot, 1) for k in hit_o},
        "spearman_full": stat(torch.cat(acc["spear"])),
    }
    return out
