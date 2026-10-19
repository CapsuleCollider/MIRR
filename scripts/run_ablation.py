"""Reproduce the ablations and the sensitivity sweeps of the read-time mechanism.

Three-view decomposition (Section 5.3).
    The paper attributes MIRR's gain to three perspectives that are switched on one at
    a time, so the ablation is a ladder rather than a list of independent removals:

    ``L0_longterm``     Stage 1 only (single state, Stage 2 off).
    ``L1_plus_recent``  + the recent window, read by a single probe (R = 1).
    ``L2_full``         + multiple interests (R = 8). This is MIRR.
    ``L2p_no_cond``     probes not conditioned on the long-term state: the read
                        becomes a fixed weighting of the window, i.e. not routing.
    ``no_route_bias``   drop the long-term bias that decides which interest wins.

Sensitivity (Section 5.4).
    ``--sweep probes`` varies R with m fixed; ``--sweep window`` varies m with R fixed.
    The two knobs are independent: R is how many interests can be separated at one
    prediction, m is how much recent behaviour is re-read.

Usage:
    python scripts/run_ablation.py --phase views --datasets Beauty ML-1M
    python scripts/run_ablation.py --phase probes --datasets Beauty
    python scripts/run_ablation.py --phase window --datasets SyntheticHard
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import recipes                                     # noqa: E402

PY = sys.executable


def run(cmd):
    print("### " + " ".join(cmd[3:]), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    print(f"### exit={r.returncode}  {time.time() - t0:.0f}s", flush=True)
    return r.returncode


def one(ds, name, cfg, seed, sub="ablation", device="cuda", extra=()):
    out = ROOT / "results" / sub / ds / f"MIRR_{name}_s{seed}.json"
    if out.exists():
        print(f"[skip] {ds}/{out.name}", flush=True)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "-m", "src.run_one", "--dataset", ds, "--model", "MIRR",
           "--seed", str(seed), "--tag", f"_{name}", "--out_dir", str(out.parent),
           "--device", device] + recipes.apply_dataset_override(
               recipes.COMMON + cfg, ds, "MIRR") + list(extra)
    return run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="views",
                    choices=["views", "probes", "window", "all"])
    ap.add_argument("--datasets", nargs="+", default=["Beauty", "ML-1M", "SyntheticHard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(recipes.SEEDS))
    ap.add_argument("--sub", default="ablation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", default="", help="comma-separated CLI args appended to "
                                                "every job (last occurrence wins)")
    args = ap.parse_args()

    extra = [x for x in args.extra.split(",") if x.strip()]
    phases = ["views", "probes", "window"] if args.phase == "all" else [args.phase]
    jobs = []
    for phase in phases:
        for ds in args.datasets:
            if phase == "views":
                for name, cfg in recipes.ABLATION.items():
                    jobs.append((ds, name, cfg))
            elif phase == "probes":
                for R in recipes.PROBE_SWEEP:
                    jobs.append((ds, f"R{R}", ["--K", "1", "--read_win", "8",
                                               "--read_probes", str(R)]))
            else:
                for m in recipes.WINDOW_SWEEP:
                    jobs.append((ds, f"m{m}", ["--K", "1", "--read_win", str(m),
                                               "--read_probes", "8"]))

    t0 = time.time()
    for ds, name, cfg in jobs:
        for seed in args.seeds:
            code = one(ds, name, cfg, seed, sub=args.sub, device=args.device,
                       extra=extra)
            if code != 0:
                print(f"[ABORT] {ds}/{name}/s{seed} failed", flush=True)
                return 1
    print(f"=== ablation done in {(time.time() - t0) / 60:.1f} min ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
