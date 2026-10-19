"""Reproduce the main table (Table 1 of the paper).

Runs every (dataset, method, seed) combination **serially** in a subprocess, so a run
that dies cannot corrupt the GPU state of the next one and memory is fully released
between jobs. Results already on disk are skipped, so an interrupted queue can simply
be restarted.

Usage:
    python scripts/run_main.py                                  # everything
    python scripts/run_main.py --datasets Beauty --models MIRR
    python scripts/run_main.py --seeds 2024 --diagnose          # + appendix diagnostics

Timing note: nine methods x five datasets x three seeds is a full sweep and takes a
long time on a single consumer GPU. Start with `--datasets Beauty --seeds 2024`.
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
# Output filename suffix per method key, so variants sharing one class do not collide.
TAGS = {"MIRR-W": "_W"}


def out_file(sub, ds, key, seed):
    return ROOT / "results" / sub / ds / f"{key}{TAGS.get(key, '')}_s{seed}.json"


def job(ds, key, seed, sub="main", extra=(), diagnose=False):
    out = out_file(sub, ds, key, seed)
    if out.exists():
        print(f"[skip] {ds}/{out.name}", flush=True)
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    cls_name, cfg = recipes.cfg_for(key, ds)
    ckpt = ROOT / "results" / "checkpoints" / ds / f"{key}{TAGS.get(key, '')}_s{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "-m", "src.run_one",
           "--dataset", ds, "--model", cls_name, "--seed", str(seed),
           "--tag", TAGS.get(key, ""), "--out_dir", str(out.parent)] + cfg + list(extra)
    if diagnose:
        cmd += ["--save_model", str(ckpt)]
    return cmd, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=recipes.DATASETS)
    ap.add_argument("--models", nargs="+", default=list(recipes.SPECS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(recipes.SEEDS))
    ap.add_argument("--sub", default="main", help="sub-directory of results/")
    ap.add_argument("--eval_ttt", action="store_true",
                    help="also evaluate the streaming test-time-training protocol")
    ap.add_argument("--diagnose", action="store_true",
                    help="after training, run the slot-collapse diagnostics")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", default="", help="extra CLI args appended to every job")
    args = ap.parse_args()

    extra = [x for x in args.extra.split(",") if x.strip()]
    if args.eval_ttt:
        extra.append("--eval_ttt")
    extra += ["--device", args.device]

    n_done = n_total = 0
    t0 = time.time()
    for ds in args.datasets:
        for key in args.models:
            for seed in args.seeds:
                n_total += 1
                built = job(ds, key, seed, sub=args.sub, extra=extra,
                            diagnose=args.diagnose)
                if built is None:
                    n_done += 1
                    continue
                cmd, ckpt = built
                print("### " + " ".join(cmd[3:]), flush=True)
                r = subprocess.run(cmd, cwd=str(ROOT))
                print(f"### exit={r.returncode}  {time.time() - t0:.0f}s elapsed",
                      flush=True)
                if r.returncode != 0:
                    print(f"[ABORT] {ds}/{key}/s{seed} failed; stopping the queue.",
                          flush=True)
                    return 1
                if args.diagnose and ckpt.exists():
                    # The diagnostics rebuild the model from the recipe, so any override
                    # used for training has to be forwarded (in the "--extra=" form, or
                    # argparse would read a leading "--" as a new option).
                    dcmd = [PY, "-u", "scripts/run_diagnose.py", "--dataset", ds,
                            "--model", key, "--seed", str(seed), "--ckpt", str(ckpt),
                            "--sub", args.sub, "--device", args.device]
                    if extra:
                        dcmd.append("--extra=" + ",".join(extra))
                    subprocess.run(dcmd, cwd=str(ROOT))
    print(f"=== done {n_done}/{n_total} (skipped) in {(time.time() - t0) / 60:.1f} min ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
