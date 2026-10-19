"""Run the slot-collapse diagnostics on an already-trained checkpoint.

Usage:
    python scripts/run_diagnose.py --dataset Beauty --model MIRR-W --seed 2024 \
        --ckpt results/checkpoints/Beauty/MIRR-W_s2024.pt

Writes ``results/<sub>/<dataset>/<model>_diag_s<seed>.json`` next to the accuracy
result produced by ``scripts/run_main.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import recipes                                            # noqa: E402
from src.analysis import slot_collapse as cd                       # noqa: E402
from src.datasets import EvalDataset, load_processed, trim_collate  # noqa: E402
from src.engine import count_params, evaluate                      # noqa: E402
from src.run_one import build_model, make_parser, set_seed          # noqa: E402


def build_subset_loader(ds, max_len, max_users, bs, seed=0, synth=False):
    """A fixed random subset of test users; diagnostics do not need the full test set."""
    _, _, test, meta = load_processed(ds)
    n = len(test)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.permutation(n)[:min(max_users, n)])
    sub = Subset(EvalDataset(test, max_len), [int(i) for i in idx])
    return DataLoader(sub, batch_size=bs, shuffle=False,
                      collate_fn=(None if synth else trim_collate)), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sub", default="main")
    ap.add_argument("--max_users", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", default="",
                    help="comma-separated CLI args overriding the recipe "
                         "(must match the configuration the checkpoint was trained with)")
    args = ap.parse_args()

    ds, key, seed = args.dataset, args.model, args.seed
    cls_name, cfg = recipes.cfg_for(key, ds)
    cfg = cfg + [x for x in args.extra.split(",") if x.strip()]
    synth = ds in recipes.SYNTH_DATASETS
    set_seed(seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ns = make_parser().parse_args(["--dataset", ds, "--model", cls_name,
                                   "--seed", str(seed), "--bs", "256"] + cfg)
    loader, meta = build_subset_loader(ds, int(ns.max_len), args.max_users, args.bs,
                                       synth=synth)
    model = build_model(cls_name, meta["n_items"], ns).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))

    te = DataLoader(EvalDataset(load_processed(ds)[2], int(ns.max_len)),
                    batch_size=args.bs, shuffle=False,
                    collate_fn=(None if synth else trim_collate))
    full = evaluate(model, te, dev, ttt=False)

    t0 = time.time()
    diag = cd.diagnose(model, loader, dev, meta["n_items"])
    res = dict(dataset=ds, model=key, seed=seed, K=int(ns.K),
               params_M=round(count_params(model), 4),
               cfg=" ".join(cfg),
               full_test={k: round(v, 5) for k, v in full.items()},
               diag=diag, diag_min=round((time.time() - t0) / 60, 2))

    out = ROOT / "results" / args.sub / ds / f"{key}_diag_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("DIAG " + json.dumps({k: res[k] for k in ("dataset", "model", "seed")}
                               | {"R_norm": diag["R_norm"],
                                  "n_eff": diag["n_eff"]["mean"],
                                  "jaccard_top50": diag["jaccard_top50"]["mean"],
                                  "dR@10": diag["delta"]["r10"]},
                               ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
