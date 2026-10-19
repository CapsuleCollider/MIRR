"""Aggregate the JSON results into Markdown and LaTeX tables.

Usage:
    python scripts/make_tables.py                       # results/main -> results/tables
    python scripts/make_tables.py --sub ablation        # also write the ablation ladder
    python scripts/make_tables.py --stdout

Every row is the mean over seeds with the sample standard deviation in parentheses;
``scripts/run_main.py --seeds 2024 2025 2026`` therefore produces three seeds per cell.
Significance of MIRR against a chosen baseline uses a paired t-test over seeds
(3 seeds -> 2 degrees of freedom), reported as the t statistic.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

METRICS = ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MRR"]

# (row label, filename glob). Order is the order of the table rows.
MAIN_ROWS = [
    ("GRU4Rec", "GRU4Rec_s*.json"),
    ("SASRec", "SASRec_s*.json"),
    ("ComiRec", "ComiRec_s*.json"),
    ("REMI", "REMI_s*.json"),
    ("TTT4Rec", "TTT4Rec_s*.json"),
    ("T^2ARec", "T^2ARec_s*.json"),
    ("DT3OR", "DT3OR_s*.json"),
    ("MIRR-W", "MIRR_W_s*.json"),
    ("MIRR", "MIRR_s*.json"),
]

# The ablation ladder: each row adds one perspective to the previous one.
ABLATION_ROWS = [
    ("L0: long-term only", "MIRR_L0_longterm_s*.json"),
    ("L1: + recent window (R=1)", "MIRR_L1_plus_recent_s*.json"),
    ("L2: + multiple interests (R=8)", "MIRR_L2_full_s*.json"),
    ("L2': probes not conditioned on u", "MIRR_L2p_no_cond_s*.json"),
    ("L2 w/o long-term interest bias", "MIRR_no_route_bias_s*.json"),
]

LATEX_NAME = {"T^2ARec": r"T$^2$ARec", "MIRR": r"\methodcell{MIRR}",
              "MIRR-W": r"\methodcell{MIRR-W}"}


def load(pattern_dir: Path, glob_pat: str):
    rows = []
    for f in sorted(glob.glob(str(pattern_dir / glob_pat))):
        try:
            rows.append(json.loads(Path(f).read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def mean_std(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / max(1, n - 1))
    return m, sd


def cell(rows, metric):
    """-> (mean, std) in percentage points."""
    vals = [100.0 * r[f"test_{metric}"] for r in rows if f"test_{metric}" in r]
    return mean_std(vals)


def by_seed(rows, metric):
    return {r["seed"]: 100.0 * r[f"test_{metric}"] for r in rows
            if f"test_{metric}" in r}


def paired_t(ours, base, metric):
    """Paired t statistic over shared seeds (relative improvement)."""
    a, b = by_seed(ours, metric), by_seed(base, metric)
    seeds = sorted(set(a) & set(b))
    d = [(a[s] - b[s]) / b[s] * 100.0 if b[s] else 0.0 for s in seeds]
    m, sd = mean_std(d)
    if len(d) < 2 or sd == 0:
        return m, float("nan"), len(d)
    return m, m / (sd / math.sqrt(len(d))), len(d)


def md_table(datasets, rows_spec, res_dir: Path, title: str):
    head = ["Method"] + [f"{ds} {m}" for ds in datasets for m in METRICS]
    out = [f"### {title}", "",
           "| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for label, pat in rows_spec:
        cells = []
        for ds in datasets:
            rows = load(res_dir / ds, pat)
            for m in METRICS:
                mu, sd = cell(rows, m)
                cells.append("--" if mu != mu else f"{mu:.3f} ({sd:.3f})")
        out.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(out)


def tex_table(datasets, rows_spec, res_dir: Path):
    """One column group per dataset, four metrics per group (no MRR: keeps it narrow)."""
    ms = [m for m in METRICS if m != "MRR"]
    cols = "l" + "|" + "|".join(["cccc"] * len(datasets))
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule"]
    lines.append("Method & " + " & ".join(
        rf"\multicolumn{{4}}{{c}}{{{ds}}}" for ds in datasets) + r" \\")
    lines.append(" & " + " & ".join(ms * len(datasets)) + r" \\")
    lines.append(r"\midrule")
    for label, pat in rows_spec:
        cells = []
        for ds in datasets:
            rows = load(res_dir / ds, pat)
            for m in ms:
                mu, sd = cell(rows, m)
                cells.append("--" if mu != mu else f"{mu:.2f}$\\pm${sd:.2f}")
        lines.append(LATEX_NAME.get(label, label).replace("T^2", "T$^2$")
                     + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def significance(res_dir: Path, datasets, baseline="T^2ARec"):
    ours_pat = dict(MAIN_ROWS)["MIRR"]
    base_pat = dict(MAIN_ROWS).get(baseline)
    out = ["### MIRR vs. " + baseline + " (paired over seeds, relative %)", "",
           "| Dataset | Metric | Rel. gain | t | n |", "|---|---|---|---|---|"]
    for ds in datasets:
        ours = load(res_dir / ds, ours_pat)
        base = load(res_dir / ds, base_pat) if base_pat else []
        for m in METRICS:
            mu, t, n = paired_t(ours, base, m)
            out.append(f"| {ds} | {m} | {mu:+.2f}% | {t:.2f} | {n} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="main")
    ap.add_argument("--datasets", nargs="+",
                    default=["Beauty", "Sports", "Toys", "ML-1M", "CDs"])
    ap.add_argument("--baseline", default="T^2ARec")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    res_dir = ROOT / "results" / args.sub
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    md = [md_table(args.datasets, MAIN_ROWS, res_dir, "Main results (%, mean over seeds)"),
          "", significance(res_dir, args.datasets, args.baseline)]
    if args.sub == "ablation" or (res_dir / args.datasets[0] / "MIRR_L0_longterm_s2024.json").exists():
        md += ["", md_table(args.datasets, ABLATION_ROWS, res_dir,
                            "Ablation ladder (%, mean over seeds)")]
    md_txt = "\n".join(md)

    tex = "% requires \\usepackage{booktabs}\n" + tex_table(args.datasets, MAIN_ROWS, res_dir)

    if args.stdout:
        print(md_txt)
        print()
        print(tex)
        return
    (out_dir / f"{args.sub}_results.md").write_text(md_txt, encoding="utf-8")
    (out_dir / f"{args.sub}_results.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {out_dir / f'{args.sub}_results.md'}")
    print(f"wrote {out_dir / f'{args.sub}_results.tex'}")


if __name__ == "__main__":
    main()
