"""Datasets and batching for leave-one-out sequential recommendation.

Splits follow the standard leave-one-out protocol produced by
``src/data/preprocess.py``:

* ``train``  = ``seq[:-2]``  (input sequence, supervised by ``seq[1:-1]``)
* ``valid``  = ``seq[:-1]``  (target ``seq[-2]``)
* ``test``   = ``seq``       (target ``seq[-1]``)

Batching.
    All methods must use the **same** batching strategy; changing it shifts results
    systematically and would confound the comparison. ``token`` (the default and the
    setting of the main table) sorts by length so that stepwise models waste few
    padded steps, and cuts batches by a *token budget* instead of a fixed sample
    count, so every gradient step sees the same number of supervised targets.
    A fixed sample count would give a batch of short sequences ~10x fewer targets
    than a batch of long ones while still costing one update, i.e. it silently
    up-weights short-history users.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_processed(name: str):
    """-> (train, valid, test, meta) for a processed dataset directory."""
    with open(PROCESSED / name / "data.pkl", "rb") as f:
        d = pickle.load(f)
    with open(PROCESSED / name / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return d["train"], d["valid"], d["test"], meta


def _trunc_pad(seq, max_len):
    s = seq[-max_len:] if len(seq) > max_len else list(seq)
    return s + [0] * (max_len - len(s)), len(s)


class TrainDataset(Dataset):
    """Input ``seq[:-1]``, target ``seq[1:]``. Right-padded; position 0 is padding."""

    def __init__(self, seqs, max_len):
        self.seqs, self.max_len = seqs, max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s, _ = _trunc_pad(list(self.seqs[i]), self.max_len)
        return (torch.tensor(s[:-1], dtype=torch.long),
                torch.tensor(s[1:], dtype=torch.long))


class EvalDataset(Dataset):
    """Input = history (target excluded), plus the target and the true history length."""

    def __init__(self, seqs, max_len):
        self.seqs, self.max_len = seqs, max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = list(self.seqs[i])
        target = s[-1]
        hist, L = _trunc_pad(s[:-1], self.max_len)
        return (torch.tensor(hist, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                L)


def trim_collate(batch):
    """Dynamic padding: crop every field to the longest real sequence in the batch.

    ``TrainDataset`` pads each sequence to ``max_len`` individually, so without this
    the time dimension is *always* ``max_len`` regardless of content: length bucketing
    then saves nothing, and a token-budget sampler would enlarge short-sequence
    batches to several hundred rows, raising peak memory instead of lowering it.
    Cropping in the collate keeps T tied to the batch content. Semantics are unchanged
    (right padding, 0 = pad).
    """
    out = default_collate(batch)
    if isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor) \
            and out[0].dim() == 2:
        M = max(int((out[0] != 0).sum(dim=1).max().item()), 1)
        out = list(out)
        for i, v in enumerate(out):
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.size(1) > M:
                out[i] = v[:, :M].contiguous()
        out = tuple(out)
    return out


class TokenBalancedSampler(torch.utils.data.Sampler):
    """Length-homogeneous batches with an approximately constant token budget."""

    def __init__(self, lengths, batch_size, seed=0, mega=50, max_scale=4.0):
        self.lengths = np.asarray(lengths)
        self.bs = batch_size
        self.seed = seed
        self.mega = mega * batch_size
        self.max_scale = max_scale
        self.epoch = 0
        L = np.maximum(self.lengths, 1)
        self.budget = batch_size * float(L.mean())
        self.max_bs = max(int(batch_size * max_scale), batch_size)
        # Number of batches stays close to N / bs because total tokens ~ N * mean_len.
        self._n_batches = max(1, int(np.ceil(float(L.sum()) / self.budget)))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths))
        L = np.maximum(self.lengths, 1)
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = idx[s:s + self.mega]
            chunk = chunk[np.argsort(L[chunk], kind="stable")]
            cur, acc = [], 0.0
            for i in chunk:
                cur.append(int(i))
                acc += L[i]
                if acc >= self.budget or len(cur) >= self.max_bs:
                    batches.append(cur)
                    cur, acc = [], 0.0
            if cur:
                batches.append(cur)
        for i in rng.permutation(len(batches)):
            yield batches[i]

    def __len__(self):
        return self._n_batches


def make_loaders(name, max_len=50, batch_size=256, eval_bs=512, num_workers=0,
                 seed=0, batching="token"):
    """batching: "shuffle" (plain shuffling) | "token" (length-homogeneous + token budget).

    **Every compared method must use the same value.**
    """
    train, valid, test, meta = load_processed(name)
    tds = TrainDataset(train, max_len)
    if batching == "token":
        lens = [min(len(s), max_len) for s in train]
        tr = DataLoader(tds, batch_sampler=TokenBalancedSampler(lens, batch_size, seed=seed),
                        num_workers=num_workers, collate_fn=trim_collate)
    elif batching == "shuffle":
        tr = DataLoader(tds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=False, collate_fn=trim_collate)
    else:
        raise ValueError(f"unknown batching={batching}")
    va = DataLoader(EvalDataset(valid, max_len), batch_size=eval_bs, shuffle=False,
                    num_workers=num_workers, collate_fn=trim_collate)
    te = DataLoader(EvalDataset(test, max_len), batch_size=eval_bs, shuffle=False,
                    num_workers=num_workers, collate_fn=trim_collate)
    return tr, va, te, meta
