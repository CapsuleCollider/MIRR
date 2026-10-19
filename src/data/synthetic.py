"""Synthetic multi-interest benchmark with a controllable interest structure.

Each synthetic user has two phases:

* **phase 1** (steps ``0 .. l1-1``): a single interest line advances a cursor around
  its item group.
* **phase 2** (steps ``l1 .. T-1``): the user jumps between ``n_lines`` active lines.
  Each line keeps its own cursor, so answering correctly requires tracking *several*
  independent cursors at once -- exactly the situation where a single fixed-size state
  runs out of capacity, and the setting the read-time router is designed for.

Difficulty is controlled by four knobs:

* ``n_lines``      — number of simultaneously active interest lines (default 3).
* ``markov_stay``  — probability of staying on the current line (default 0.5, i.e. the
  next line is unpredictable while the next item *given* the line is deterministic).
  ``>= 1`` is a sentinel meaning strict round-robin, which makes the task solvable by a
  fixed rule and saturates every method.
* ``n_groups`` / ``group_size`` — catalog size (default 8 x 100 = 800 items).
* ``T``            — sequence length (default 60).

Splitting is by **user**, and the switch point is drawn per user, so a model cannot
exploit absolute positions: every position is seen during training.

Usage:
    python -m src.data.synthetic --preset hard
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _make_user(rng, n_groups, group_size, T, l1, n_lines, markov_stay):
    """One user sequence. Returns (seq, l1)."""
    strict = markov_stay >= 1.0                     # sentinel: strict round-robin
    gs = [int(g) for g in rng.choice(n_groups, size=n_lines, replace=False)]
    c = {g: int(rng.randint(group_size)) for g in gs}
    s = []
    for _ in range(l1):                             # phase 1: one line
        g = gs[0]
        s.append(g * group_size + c[g] + 1)
        c[g] = (c[g] + 1) % group_size
    cur = 0
    while len(s) < T:                               # phase 2: n_lines lines
        if strict:
            j = (cur + 1) % n_lines
        elif rng.rand() < markov_stay:
            j = cur
        else:
            others = [k for k in range(n_lines) if k != cur] or [cur]
            j = others[int(rng.randint(len(others)))]
        cur = j
        g = gs[j]
        s.append(g * group_size + c[g] + 1)
        c[g] = (c[g] + 1) % group_size
    return s, l1


PRESETS = {
    # name: kwargs for generate()
    "easy": dict(out_name="SyntheticEasy", n_groups=4, group_size=100, T=40,
                 l1_range=(12, 24), n_lines=2, markov_stay=0.5),
    "hard": dict(out_name="SyntheticHard", n_groups=8, group_size=100, T=60,
                 l1_range=(15, 30), n_lines=3, markov_stay=0.5),
    "harder": dict(out_name="SyntheticHarder", n_groups=10, group_size=150, T=80,
                   l1_range=(20, 40), n_lines=4, markov_stay=0.5),
}


def generate(n_train=4000, n_valid=1000, n_test=1000, n_groups=4, group_size=100,
             T=40, l1_range=(12, 24), n_lines=2, markov_stay=1e9, seed=2024,
             out_name="SyntheticV2"):
    """Write ``data/processed/<out_name>/{data.pkl, meta.json}`` and return the stats."""
    rng = np.random.RandomState(seed)
    n_items = n_groups * group_size + 1
    lo_l1, hi_l1 = l1_range

    def batch(n):
        out = []
        for _ in range(n):
            l1 = int(rng.randint(lo_l1, hi_l1 + 1))
            s, _l = _make_user(rng, n_groups, group_size, T, l1, n_lines, markov_stay)
            out.append(s)
        return out

    train = batch(n_train)                          # training users: the full T steps
    valid = batch(n_valid)
    test = batch(n_test)

    # Per-user switch segment: from just after that user's switch point to T-2. The
    # first item of a newly entered line is skipped because its cursor has never been
    # observed and is therefore unpredictable.
    test_switch = [s[:t + 1] for s in test for t in range(lo_l1 + 1, T - 1)]

    stats = dict(dataset=out_name, n_train_users=n_train, n_valid_users=n_valid,
                 n_test_users=n_test, n_items=n_items, n_groups=n_groups,
                 group_size=group_size, seq_len=T, l1_range=list(l1_range),
                 n_lines=n_lines,
                 markov_stay=("strict" if markov_stay >= 1.0 else markov_stay),
                 switch_range="per-user [l1+1, T-2]", n_train=len(train),
                 n_valid=len(valid), n_test=len(test), n_test_switch=len(test_switch))
    out = PROCESSED / out_name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "data.pkl", "wb") as f:
        pickle.dump(dict(train=train, valid=valid, test=test,
                         test_switch=test_switch), f)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(stats)
    return stats


def check_rule(name="SyntheticHard", topk=10):
    """Oracle of this dataset: put every active line's next item in the top-K.

    Should be ~1.0 on both ``test`` and ``test_switch``; anything lower means the
    generator is wrong or K is too small. Cursors are recovered from the history only,
    never from the target.
    """
    with open(PROCESSED / name / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    gs_ = meta["group_size"]
    with open(PROCESSED / name / "data.pkl", "rb") as f:
        d = pickle.load(f)

    def oracle_hit(seqs):
        ok = 0
        for e in seqs:
            e = [int(x) for x in e]
            tgt, hist = e[-1], e[:-1]
            nxt = {}
            for x in hist:
                if x == 0:
                    continue
                g, k = (x - 1) // gs_, (x - 1) % gs_
                nxt[g] = g * gs_ + (k + 1) % gs_ + 1
            ok += int(tgt in list(nxt.values())[:topk])
        return ok / len(seqs)

    for key in ("test", "test_switch"):
        print(f"{key}: n={len(d[key])}  oracle R@{topk} = {oracle_hit(d[key]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="hard", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    kw = dict(PRESETS[a.preset])
    name = kw["out_name"]
    generate(seed=a.seed, **kw)
    check_rule(name)


if __name__ == "__main__":
    main()
