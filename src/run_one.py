"""Single-run entry point: train -> evaluate -> save one JSON result.

Usage:
    python -m src.run_one --dataset Beauty --model MIRR --seed 2024 \
        --out_dir results/main/Beauty

Hyper-parameters are normally supplied by :mod:`src.recipes` through
``scripts/run_main.py``; passing them by hand is supported for sweeps.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .datasets import make_loaders
from .engine import count_params, evaluate, measure_latency, train_model
from .models.baselines import ComiRec, GRU4Rec, REMI, SASRec
from .models.mirr import MIRR
from .models.online import DT3OR, T2ARec, TTT4Rec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Test-time learning rate grid for the TTT baselines. Each method searches its own
# step size on validation data, so MIRR is never compared against an under-tuned
# adaptation step.
TTT_LR_GRID = [1e-5, 1e-4, 1e-3]


def safe_name(s: str) -> str:
    """Make an ablation tag safe as a filename (tags may contain '/')."""
    return s.replace("/", "-").replace("\\", "-").replace(":", "-")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_model(name, n_items, args):
    common = dict(d_model=args.d, max_len=args.max_len, n_layers=args.n_layers,
                  dropout=args.dropout)
    if name == "GRU4Rec":
        return GRU4Rec(n_items, **common)
    if name == "SASRec":
        return SASRec(n_items, n_heads=args.n_heads, **common)
    if name == "ComiRec":
        m = ComiRec(n_items, n_heads=args.n_heads, K=args.K, **common)
        m.n_sup_pos = args.n_sup_pos
        return m
    if name == "REMI":
        m = REMI(n_items, n_heads=args.n_heads, K=args.K, **common)
        m.n_sup_pos = args.n_sup_pos
        return m
    if name == "TTT4Rec":
        return TTT4Rec(n_items, **common)
    if name == "T^2ARec":
        return T2ARec(n_items, align_w=args.align_w, **common)
    if name == "DT3OR":
        return DT3OR(n_items, K=args.K, align_w=args.align_w, sd_w=args.sd_w, **common)
    if name == "MIRR":
        return MIRR(n_items, K=args.K, read_win=args.read_win,
                    read_probes=args.read_probes, read_topm=args.read_topm,
                    read_cond=args.read_cond, per_user_alpha=args.per_user_alpha,
                    route_bias=args.route_bias, bptt=args.bptt, **common)
    raise ValueError(name)


def make_parser():
    p = argparse.ArgumentParser()
    # ---- experiment ---- #
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--tag", default="",
                   help="suffix of the output filename; used to distinguish variants "
                        "that share a model class (e.g. MIRR vs. MIRR-W)")
    p.add_argument("--out_dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_model", default="")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no_latency", action="store_true")
    p.add_argument("--eval_ttt", action="store_true",
                   help="also run the streaming test-time-training protocol")
    # ---- shared optimisation ---- #
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--eval_bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--sched", default="none", choices=["none", "cosine", "plateau"])
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--batching", default="token", choices=["shuffle", "token"])
    # ---- baselines ---- #
    p.add_argument("--n_sup_pos", type=int, default=1,
                   help="ComiRec/REMI: sampled supervision positions per sequence")
    p.add_argument("--align_w", type=float, default=0.1)
    p.add_argument("--sd_w", type=float, default=0.1)
    p.add_argument("--ttt_lr", type=float, default=1e-3)
    p.add_argument("--ttt_lr_auto", type=int, default=1,
                   help="1 = search the test-time learning rate on the validation set")
    # ---- MIRR ---- #
    p.add_argument("--read_win", type=int, default=8,
                   help="m: recent interactions kept outside the state and re-read")
    p.add_argument("--read_probes", type=int, default=8,
                   help="R: number of interest probes")
    p.add_argument("--read_topm", type=int, default=-1,
                   help="M: candidate-list size; -1 selects max(512, ceil(0.04|I|))")
    p.add_argument("--read_cond", type=int, default=1,
                   help="1 = condition the probes on the long-term state")
    p.add_argument("--per_user_alpha", type=int, default=0,
                   help="1 = sample-dependent fusion weight")
    p.add_argument("--route_bias", type=int, default=1,
                   help="1 = let the long-term state bias which interest wins")
    p.add_argument("--bptt", type=int, default=0,
                   help="truncated BPTT interval in steps (0 = no truncation)")
    return p


def _auto_bs(args, n_items):
    """Shrink the batch size on large catalogs so that (B, T, N) logits fit in memory.

    On CDs (64k items) a batch of 256 x 50 already produces several GB of logits before
    the backward pass. With dynamic padding the time dimension follows the batch
    content, so the thresholds below are far more permissive than a fixed-T estimate.
    """
    bs, ebs = args.bs, args.eval_bs
    if n_items > 40000:
        bs, ebs = min(bs, 128), min(ebs, 128)
    elif n_items > 20000:
        bs, ebs = min(bs, 256), min(ebs, 256)
    # ComiRec scores K interests per sampled position, so its score tensor is K times
    # larger than everyone else's; on the largest catalog it needs a smaller batch.
    if args.model == "ComiRec" and n_items > 40000:
        bs = min(bs, 32)
    if (bs, ebs) != (args.bs, args.eval_bs):
        print(f"  [auto] n_items={n_items} -> bs={bs}, eval_bs={ebs}", flush=True)
    return bs, ebs


def select_ttt_lr(model, valid_loader, device, grid=TTT_LR_GRID, log=print):
    """Grid-search the test-time learning rate on the validation set."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    scores = {}
    for lr in grid:
        model.load_state_dict(sd)
        scores[lr] = evaluate(model, valid_loader, device, ttt=True, ttt_lr=lr)["NDCG@10"]
    model.load_state_dict(sd)
    best = max(scores, key=scores.get)
    log("  [ttt_lr] " + "  ".join(f"{lr:.0e}:{v * 100:.3f}" for lr, v in scores.items())
        + f"  -> {best:.0e}")
    return best, {f"{k:.0e}": v for k, v in scores.items()}


def run_one(args, log=print):
    set_seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # First pass only to read n_items (loaders are cheap; the data is cached).
    _, _, _, meta0 = make_loaders(args.dataset, max_len=args.max_len,
                                  batch_size=args.bs, eval_bs=args.eval_bs,
                                  seed=args.seed, batching=args.batching)
    bs, ebs = _auto_bs(args, meta0["n_items"])
    tr, va, te, meta = make_loaders(args.dataset, max_len=args.max_len, batch_size=bs,
                                    eval_bs=ebs, seed=args.seed, batching=args.batching)
    model = build_model(args.model, meta["n_items"], args).to(dev)
    n_params = count_params(model)

    # Synthetic datasets report n_test_users rather than n_users.
    n_users = meta.get("n_users", meta.get("n_test_users", 0))
    print(f"\n### {args.dataset} | {args.model}{args.tag} | seed={args.seed} | "
          f"items={meta['n_items']} users={n_users} params={n_params:.3f}M", flush=True)
    t0 = time.time()
    model, best, best_ep, _ = train_model(
        model, tr, va, dev, epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
        patience=args.patience, sched=args.sched, warmup=args.warmup,
        log=(lambda s: (None if args.quiet else print(s, flush=True))))
    train_min = (time.time() - t0) / 60

    res = dict(dataset=args.dataset, model=args.model, run_name=args.model + args.tag,
               seed=args.seed, tag=args.tag, K=args.K, params_M=round(n_params, 4),
               best_epoch=best_ep, best_valid_ndcg10=round(float(best), 5),
               train_total_min=round(train_min, 2),
               train_min_per_epoch=round(train_min / max(best_ep, 1), 3),
               n_items=meta["n_items"], n_users=n_users)

    # Static evaluation: the protocol of the main table (no test-time update).
    m_static = evaluate(model, te, dev, ttt=False)
    res.update({f"test_{k}": round(v, 5) for k, v in m_static.items()})

    if dev.type == "cuda":
        res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
        torch.cuda.reset_peak_memory_stats()

    # Streaming test-time-training protocol (Section 5.5).
    if args.eval_ttt and getattr(model, "is_ttt", False):
        if args.ttt_lr_auto:
            args.ttt_lr, grid = select_ttt_lr(model, va, dev, log=log)
            res["ttt_lr"] = args.ttt_lr
            res["ttt_lr_grid_ndcg10"] = grid
        m_ttt = evaluate(model, te, dev, ttt=True, ttt_lr=args.ttt_lr)
        res.update({f"ttt_{k}": round(v, 5) for k, v in m_ttt.items()})

    if not args.no_latency:
        res["latency_ms"] = round(measure_latency(model, te, dev, ttt=False), 4)
        if args.eval_ttt and getattr(model, "is_ttt", False):
            lat_t = measure_latency(model, te, dev, ttt=True, ttt_lr=args.ttt_lr)
            res["latency_ttt_ms"] = round(lat_t, 4)
            res["ttt_overhead_ms"] = round(lat_t - res["latency_ms"], 4)

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS / "raw" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = safe_name(f"{args.model}{args.tag}_s{args.seed}")
    with open(out_dir / f"{fname}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    return res


def main():
    run_one(make_parser().parse_args())


if __name__ == "__main__":
    main()
