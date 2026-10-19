"""Preprocessing of the public benchmarks (Amazon 2018 / MovieLens-1M).

Pipeline (identical for every dataset):

1. read the raw file and keep implicit feedback;
2. iteratively apply 5-core filtering (both users and items need >= 5 interactions);
3. group by user and sort by timestamp to build one sequence per user;
4. leave-one-out split: ``train = seq[:-2]``, valid target ``seq[-2]``, test target ``seq[-1]``;
5. re-index items to ``1..N`` with ``0`` reserved for padding.

Output: ``data/processed/<name>/{data.pkl, meta.json}``.

Usage:
    python -m src.data.preprocess --datasets Beauty Sports Toys CDs ML-1M
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

AMAZON_FILES = {
    "Beauty": "ratings_Beauty.csv",
    "Sports": "ratings_Sports_and_Outdoors.csv",
    "Toys": "ratings_Toys_and_Games.csv",
    "CDs": "ratings_CDs_and_Vinyl.csv",
}


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_amazon(path: Path, min_rating: float = 0.0):
    """Amazon 2018 ratings-only: ``user,item,rating,timestamp`` (no header)."""
    users, items, ts = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 4:
                continue
            try:
                r = float(p[2])
            except ValueError:
                continue
            if r < min_rating:
                continue
            users.append(p[0])
            items.append(p[1])
            ts.append(int(float(p[3])))
    return users, items, ts


def read_ml1m(path: Path, min_rating: float = 0.0):
    """MovieLens-1M ``ratings.dat``: ``UserID::MovieID::Rating::Timestamp``."""
    users, items, ts = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("::")
            if len(p) < 4:
                p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            try:
                r = float(p[2])
            except ValueError:
                continue
            if r < min_rating:
                continue
            users.append(p[0])
            items.append(p[1])
            ts.append(int(float(p[3])))
    return users, items, ts


# --------------------------------------------------------------------------- #
# 5-core filtering and sequence construction
# --------------------------------------------------------------------------- #
def k_core_filter(users, items, ts, k=5, max_iter=20):
    """Iteratively drop users and items with fewer than ``k`` interactions."""
    u_arr = np.array(users, dtype=object)
    i_arr = np.array(items, dtype=object)
    t_arr = np.array(ts, dtype=np.int64)
    alive = np.ones(len(users), dtype=bool)
    for _ in range(max_iter):
        ucnt, icnt = defaultdict(int), defaultdict(int)
        for idx in np.nonzero(alive)[0]:
            ucnt[u_arr[idx]] += 1
            icnt[i_arr[idx]] += 1
        mask_u = np.array([ucnt[u] >= k for u in u_arr])
        mask_i = np.array([icnt[i] >= k for i in i_arr])
        new_alive = alive & mask_u & mask_i
        if new_alive.sum() == alive.sum():
            break
        alive = new_alive
    return u_arr[alive], i_arr[alive], t_arr[alive]


def build_sequences(u_arr, i_arr, t_arr, min_len=5):
    """Group by user, sort by time (ties broken by item id), and re-index items."""
    order = np.lexsort((i_arr.astype(str), t_arr, u_arr.astype(str)))
    u_arr, i_arr = u_arr[order], i_arr[order]

    imap, seqs = {}, []
    cur_user, cur = None, []
    for u, it in zip(u_arr, i_arr):
        if u != cur_user:
            if cur_user is not None and len(cur) >= min_len:
                seqs.append(cur)
            cur_user, cur = u, []
        cur.append(it)
    if cur_user is not None and len(cur) >= min_len:
        seqs.append(cur)

    for s in seqs:                                  # items -> 1..N, 0 = padding
        for it in s:
            if it not in imap:
                imap[it] = len(imap) + 1
    return [[imap[it] for it in s] for s in seqs], imap


def leave_one_out(seqs):
    """train = seq[:-2]; valid input seq[:-1] (target seq[-2]); test input seq (target seq[-1])."""
    train, valid, test = [], [], []
    for s in seqs:
        if len(s) < 3:
            continue
        train.append(s[:-2])
        valid.append(s[:-1])
        test.append(s)
    return train, valid, test


def process(name: str, reader, path: Path, min_rating=0.0, k_core=5, seed=2024):
    print(f"[{name}] reading {path} ...")
    users, items, ts = reader(path, min_rating)
    print(f"[{name}] raw interactions: {len(users):,}")
    users, items, ts = k_core_filter(users, items, ts, k=k_core)
    print(f"[{name}] after {k_core}-core: {len(users):,}")
    seqs, imap = build_sequences(users, items, ts)
    print(f"[{name}] users={len(seqs):,}  items={len(imap):,}")

    random.Random(seed).shuffle(seqs)
    train, valid, test = leave_one_out(seqs)
    lens = np.array([len(s) for s in seqs])
    stats = dict(
        dataset=name,
        n_users=len(seqs),
        n_items=len(imap) + 1,
        n_interactions=int(lens.sum()),
        avg_len=float(lens.mean()),
        max_len=int(lens.max()),
        n_train=len(train), n_valid=len(valid), n_test=len(test),
        density=float(lens.sum() / (len(seqs) * (len(imap) + 1))),
    )
    out_dir = PROCESSED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump(dict(train=train, valid=valid, test=test), f)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[{name}] {stats}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Beauty", "Sports", "Toys", "CDs", "ML-1M"])
    ap.add_argument("--k_core", type=int, default=5)
    ap.add_argument("--min_rating", type=float, default=0.0)
    args = ap.parse_args()

    all_stats = {}
    for name in args.datasets:
        if name in AMAZON_FILES:
            p = RAW / AMAZON_FILES[name]
            if not p.exists():
                print(f"[skip] {name}: {p} not found")
                continue
            all_stats[name] = process(name, read_amazon, p, args.min_rating, args.k_core)
        elif name in ("ML-1M", "ml-1m"):
            p = RAW / "ml-1m" / "ratings.dat"
            if not p.exists():
                p = RAW / "ml-1m" / "ml-1m" / "ratings.dat"
            if not p.exists():
                print(f"[skip] ML-1M: ratings.dat not found under {RAW / 'ml-1m'}")
                continue
            all_stats["ML-1M"] = process(name, read_ml1m, p, args.min_rating, args.k_core)
        else:
            print(f"[skip] unknown dataset {name}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print("\n=== summary ===")
    for k, v in all_stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
