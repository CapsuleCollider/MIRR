"""Training and evaluation engine.

Two evaluation protocols are implemented:

* :func:`evaluate_static` — a single forward pass per sequence. Used by every method
  in the main table, including the test-time-training baselines, because the main
  table reports accuracy *without* a test-time parameter update.
* :func:`evaluate` with ``ttt=True`` — the streaming protocol: the model reads the
  sequence step by step, emits a prediction at the position whose target is the
  held-out item, and after every other position takes one self-supervised update on
  the *already observed* next interaction. The held-out target never enters an update.
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from .models.common import metrics_from_logits


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def train_loss_batch(model, inp, tgt, device):
    """Returns ``(loss, n_targets)``.

    MIRR accumulates its own position-wise loss inside :meth:`MIRR.forward` (it needs
    the stepwise window), so it returns a sum and a target count here. ComiRec / REMI
    are supervised at ``S`` sampled positions per sequence (see
    :mod:`src.models.baselines`); every other model is supervised at all positions.
    """
    if model.__class__.__name__ == "MIRR":
        loss_sum, n_tgt, _ = model.forward(inp)
        if n_tgt == 0:
            return inp.new_zeros((), dtype=torch.float32), 0
        return loss_sum / n_tgt, n_tgt

    if getattr(model, "train_on_final", False):          # ComiRec / REMI
        # Multi-position supervision: sample S positions per sequence. Under the causal
        # mask h[:, t] is the encoding of the prefix 0..t, so masking everything after t
        # inside the interest extraction gives the interest vector of position t without
        # re-encoding and without duplicating users. This is an unbiased estimate of the
        # full-position loss.
        S = int(getattr(model, "n_sup_pos", 1) or 1)
        if S > 1 and hasattr(model, "forward_multi_pos"):
            B, T = inp.shape
            L = (inp != 0).sum(dim=1)                            # (B,) true lengths
            nval = (L - 1).clamp(min=1).float()                  # supervisable positions
            pos = (torch.rand(B, S, device=device) * nval.unsqueeze(1)).long()
            pos = pos.clamp(max=max(T - 2, 0))                   # (B, S)
            y = inp[torch.arange(B, device=device).unsqueeze(1),
                    (pos + 1).clamp(max=T - 1)]                  # (B, S)
            keep = (L >= 2).unsqueeze(1).expand(B, S) & (y != 0)
            if not bool(keep.any()):
                return inp.new_zeros((), dtype=torch.float32), 0
            # The score tensor is (rows, K, n_items) with rows = B*S; chunk the S
            # dimension to respect the memory budget.
            budget = int(os.environ.get("MIRR_SUP_POS_ROWS", "4096"))
            chunk = max(1, min(S, budget // max(B, 1)))
            outs, ys = [], []
            for c0 in range(0, S, chunk):
                sl = slice(c0, c0 + chunk)
                k = keep[:, sl].reshape(-1)
                if not bool(k.any()):
                    continue
                outs.append(model.forward_multi_pos(inp, pos[:, sl])[k])
                ys.append(y[:, sl].reshape(-1)[k])
            logits = torch.cat(outs, dim=0)
            yt = torch.cat(ys, dim=0)
            loss = F.cross_entropy(logits, yt)
            reg = model.reg_loss(inp) if hasattr(model, "reg_loss") else 0.0
            return loss + reg, int(yt.numel())
        # Original single-position supervision: drop the last real item from the input
        # and predict it from the second-to-last real position.
        B = inp.size(0)
        rows = torch.arange(B, device=device)
        idx = (inp != 0).sum(dim=1).clamp(min=2) - 1
        inp2 = inp.clone()
        inp2[rows, idx] = 0
        logits = model.final_logits(inp2)
        y = tgt[rows, idx - 1]
        loss = F.cross_entropy(logits, y)
        reg = model.reg_loss(inp2) if hasattr(model, "reg_loss") else 0.0
        return loss + reg, B

    logits = model(inp)                                   # (B, T, n_items)
    y = tgt
    m = (y != 0)
    if m.sum() == 0:
        return inp.new_zeros((), dtype=torch.float32), 0
    loss = F.cross_entropy(logits[m], y[m])
    reg = model.reg_loss(inp) if hasattr(model, "reg_loss") else 0.0
    return loss + reg, int(m.sum())


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_static(model, loader, device, ks=(10, 20)):
    """Full-catalog ranking metrics for a frozen model."""
    model.eval()
    allm = []
    for batch in loader:
        inp, target = batch[0].to(device), batch[1].to(device)
        allm.append(metrics_from_logits(model.final_logits(inp), target, ks))
    keys = allm[0].keys()
    return {k: sum(d[k] for d in allm) / len(allm) for k in keys}


class _masked_cache:
    """Temporarily replace ``model._last_u`` / ``_last_H`` by a masked subset.

    Test-time closures read these whole-batch buffers, but the loss is computed on the
    subset of rows whose next interaction is real.
    """

    def __init__(self, model, m, B):
        self.model, self.m, self.B = model, m, B
        self.saved = {}

    def __enter__(self):
        for k in ("_last_u", "_last_H"):
            v = getattr(self.model, k, None)
            if isinstance(v, torch.Tensor) and v.shape[0] == self.B \
                    and int(self.m.sum()) != self.B:
                self.saved[k] = v
                setattr(self.model, k, v[self.m])
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            setattr(self.model, k, v)
        return False


def _trunc_bptt(model, state):
    """Detach every stepwise buffer. Must go through ``model.detach_state``: MIRR also
    carries the recent-interaction window, and leaving it attached makes the next
    ``backward()`` fail."""
    if hasattr(model, "detach_state"):
        return model.detach_state(state)
    return [s.detach() if torch.is_tensor(s) else s for s in state]


def evaluate(model, loader, device, ks=(10, 20), ttt=False, ttt_lr=1e-3):
    """Unified evaluation. ``ttt=False`` is the static protocol used in the main table.

    ``ttt=True`` restores the trained weights and rebuilds the optimizer before every
    batch, so adaptation is confined to "the recent behaviour of the users in this
    batch" and never accumulates across batches.
    """
    model.eval()
    if not ttt:
        return evaluate_static(model, loader, device, ks)

    params = model.ttt_param_groups()
    sd0 = {k: v.detach().clone() for k, v in model.state_dict().items()} if params else None
    allm = []
    for batch in loader:
        opt = torch.optim.Adam(params, lr=ttt_lr) if params else None
        if sd0 is not None:
            model.load_state_dict(sd0)
        inp, target = batch[0].to(device), batch[1].to(device)
        lengths = torch.as_tensor(batch[2], device=device)
        B, T = inp.shape
        state = model.init_state(B, device)
        preds = torch.zeros(B, model.n_items, device=device)
        for t in range(T):
            logits, state, info = model.step(inp[:, t], state, pos=t)
            done = (lengths == t + 1)
            if done.any():
                preds[done] = logits[done]
            if opt is not None and t + 1 < T:
                y = inp[:, t + 1]
                m = (y != 0)
                if m.any():
                    with _masked_cache(model, m, B):
                        loss = model.ttt_closure(logits[m], y[m], info, model)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    state = _trunc_bptt(model, state)
        allm.append(metrics_from_logits(preds, target, ks))
    if sd0 is not None:
        model.load_state_dict(sd0)
    keys = allm[0].keys()
    return {k: sum(d[k] for d in allm) / len(allm) for k in keys}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _make_sched(opt, sched, epochs, warmup):
    """Learning-rate schedule.

    ``none``    -- constant learning rate (the setting reported in the paper).
    ``cosine``  -- linear warmup then cosine decay to 0.
    ``plateau`` -- halve the learning rate when validation NDCG@10 stalls.
    """
    if sched == "none" or epochs <= 1:
        return None
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(1, warmup or 3))
    if sched == "cosine":
        w = max(0, int(warmup))

        def fn(step):                       # step counts from 0 to epochs
            if w > 0 and step < w:
                return float(step + 1) / float(w)
            p = float(step - w) / float(max(1, epochs - w))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

        return torch.optim.lr_scheduler.LambdaLR(opt, fn)
    raise ValueError(f"unknown sched={sched}")


def train_model(model, train_loader, valid_loader, device, epochs=60, lr=1e-3,
                weight_decay=0.0, patience=8, ks=(10, 20), log=print,
                max_minutes=None, sched="none", warmup=0):
    """Train with Adam, select the best checkpoint by validation NDCG@10.

    Early stopping uses patience on validation NDCG@10; the best state is restored at
    the end. Returns ``(model, best_ndcg10, best_epoch, history)``.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = _make_sched(opt, sched, epochs, warmup)
    best, best_state, best_epoch = -1.0, None, -1
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            inp, tgt = batch[0].to(device), batch[1].to(device)
            loss, n = train_loss_batch(model, inp, tgt, device)
            if n == 0:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        m = evaluate_static(model, valid_loader, device, ks)
        if sch is not None:
            if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step(m["NDCG@10"])
            else:
                sch.step()
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
        score = m["NDCG@10"]
        if score > best:
            best, best_epoch = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  ep{ep:3d} loss={tot / max(nb, 1):.4f} valid NDCG@10={m['NDCG@10']:.4f} "
            f"Recall@10={m['Recall@10']:.4f}")
        if ep - best_epoch >= patience:
            log(f"  early stop @ epoch {ep} (best {best_epoch})")
            break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            log(f"  time budget reached @ epoch {ep}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_epoch, hist


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_latency(model, loader, device, ttt=False, ttt_lr=1e-3,
                    warmup=1, reps=1, max_batches=8):
    """Milliseconds per sample (first ``max_batches`` batches only)."""
    model.eval()
    total, n = 0.0, 0
    for _ in range(warmup):
        with torch.no_grad():
            for batch in loader:
                model.final_logits(batch[0].to(device))
                break
    nb = 0
    for _ in range(reps):
        for batch in loader:
            if nb >= max_batches:
                break
            nb += 1
            inp = batch[0].to(device)
            t0 = time.perf_counter()
            if not ttt:
                with torch.no_grad():
                    model.final_logits(inp)
            else:
                B, T = inp.shape
                state = model.init_state(B, device)
                params = model.ttt_param_groups()
                sd = {k: v.detach().clone() for k, v in model.state_dict().items()} \
                    if params else None
                opt = torch.optim.Adam(params, lr=ttt_lr) if params else None
                for t in range(T):
                    logits, state, info = model.step(inp[:, t], state, pos=t)
                    if opt is not None and t + 1 < T:
                        y = inp[:, t + 1]
                        m = y != 0
                        if m.any():
                            with _masked_cache(model, m, B):
                                loss = model.ttt_closure(logits[m], y[m], info, model)
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            opt.step()
                            state = _trunc_bptt(model, state)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total += (time.perf_counter() - t0) * 1000
            n += inp.size(0)
    return total / max(n, 1)
