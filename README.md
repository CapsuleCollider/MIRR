# MIRR: Multi-Interest Read-Time Router

Official implementation of **MIRR**, a sequential recommender that defers multi-interest
routing from *write time* to *read time*.

> Existing multi-interest recommenders assign an interaction to an interest when it is
> written into the user state. At that moment only the past is visible, and once the
> assignment is merged into a compressed state it can never be separated again. MIRR
> keeps a **single** state for long-term preference, keeps the most recent interactions
> **outside** the state, and decides which interests are active only when a prediction is
> requested. Recent adaptation therefore comes from *what is read*, not from a gradient
> step: no parameter is updated at serving time.

## Method in one figure

```
                       ┌──────────────────────── Stage 1 (long term) ─────────────┐
  x_1 … x_{t-m}  ─────▶│  single-state gated SSM  ──▶  u_t  ──▶  full-catalog      │
                       │                                          scores  base(i)  │
                       └──────────────────────────────┬───────────────────────────┘
                                                      │ top-M candidates C_t
                       ┌──────────────────────── Stage 2 (read time) ─────────────┐
  x_{t-m+1} … x_t  ───▶│  R probes (conditioned on u_t) attend over the window      │
   (never merged)      │  ──▶  U_1 … U_R  ──▶  l(i) = max_r ( l_r(i) + λ·b_r )     │
                       └──────────────────────────────┬───────────────────────────┘
                                                      ▼
                    score(i) = (1-α)·base(i) + α·l(i)   for i ∈ C_t,  base(i) otherwise
```

Nothing produced by Stage 2 is written back into the state, so a wrong grouping at step
`t` cannot affect step `t+1`.

## Repository layout

```
MIRR/
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt           torch + numpy only
├── src/
│   ├── data/
│   │   ├── preprocess.py      5-core filtering + leave-one-out split for the public
│   │   │                      benchmarks -> data/processed/<name>/{data.pkl, meta.json}
│   │   └── synthetic.py       generator for the synthetic multi-interest benchmark with
│   │                          controllable difficulty; also its oracle check
│   ├── datasets.py            TrainDataset / EvalDataset, dynamic-padding collate, and
│   │                          the token-balanced sampler shared by all methods
│   ├── engine.py              training loop, static evaluation, streaming (TTT)
│   │                          evaluation, latency measurement
│   ├── models/
│   │   ├── common.py          item embedding, full-catalog ranking metrics, base interface
│   │   ├── ssm.py             gated diagonal selective SSM (parallel scan + online step)
│   │   ├── baselines.py       GRU4Rec, SASRec, ComiRec, REMI  (static)
│   │   ├── online.py          shared online backbone + TTT4Rec, T^2ARec, DT3OR
│   │   └── mirr.py            MIRR (read-time) and MIRR-W (write-time ablation)
│   ├── analysis/
│   │   └── slot_collapse.py   appendix diagnostics: R_norm, n_eff, jaccard@50,
│   │                          load share, and the mean-substitution test
│   ├── recipes.py             hyper-parameters of every reported run (single source
│   │                          of truth for the tables)
│   └── run_one.py             entry point for one (dataset, method, seed) run
├── scripts/
│   ├── prepare_data.sh        preprocess the public benchmarks + generate SyntheticHard
│   ├── run_main.py            reproduce the main table (serial queue, skips finished runs)
│   ├── run_ablation.py        ablation ladder + R / m sensitivity sweeps
│   ├── run_diagnose.py        slot-collapse diagnostics on a saved checkpoint
│   └── make_tables.py         aggregate results/<sub>/*.json into Markdown and LaTeX
├── data/
│   ├── raw/README.md          where to download each raw file
│   └── processed/             generated; not committed
└── results/                   generated; not committed
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with PyTorch 2.x (CPU and CUDA). No other dependency is required; the code does
not use `mamba-ssm` (see the note in `src/models/ssm.py`).

## Data preparation

```bash
bash scripts/prepare_data.sh
```

1. Download the raw files listed in `data/raw/README.md` into `data/raw/`
   (Amazon 2018 ratings-only, MovieLens-1M).
2. Run the script. It applies 5-core filtering, builds one chronological sequence per
   user, and writes the leave-one-out splits. `SyntheticHard` is generated locally and
   needs no download.

Statistics of the processed datasets (5-core, leave-one-out):

| Dataset | Users | Items | Avg. length | Density | M (auto) |
| --- | --- | --- | --- | --- | --- |
| Beauty | 22,363 | 12,102 | 8.88 | 0.00073 | 512 |
| Sports | 35,598 | 18,358 | 8.32 | 0.00045 | 735 |
| Toys | 19,412 | 11,925 | 8.63 | 0.00072 | 512 |
| ML-1M | 6,040 | 3,417 | 165.50 | 0.04843 | 512 |
| CDs | 40,000 | 64,444 | 14.44 | 0.00022 | 2578 |

`M` is the candidate-list size selected by the rule `M = max(512, ceil(0.04·|I|))`.

## Reproducing the main table

```bash
# everything: 5 datasets x 9 methods x 3 seeds  (long; start small)
python scripts/run_main.py

# a quick sanity run first
python scripts/run_main.py --datasets Beauty --models MIRR SASRec --seeds 2024

# aggregate into results/tables/main_results.{md,tex}
python scripts/make_tables.py
```

Jobs run **serially in a subprocess**: memory is released between runs and a failure
cannot corrupt the next one. Finished runs are detected by their output file, so an
interrupted queue can simply be restarted.

Results land in `results/main/<dataset>/<method>[_tag]_s<seed>.json` and contain the
full-catalog `test_*` metrics, parameter count, best epoch, training time, peak memory
and latency.

### Methods

| Key | Class | Group |
| --- | --- | --- |
| `GRU4Rec`, `SASRec` | `GRU4Rec`, `SASRec` | single-interest, fixed-state |
| `ComiRec`, `REMI` | `ComiRec`, `REMI` | write-time multi-interest |
| `TTT4Rec`, `T^2ARec`, `DT3OR` | online | test-time training |
| `MIRR-W` | `MIRR` | write-time multi-slot ablation (ours, negative control) |
| `MIRR` | `MIRR` | **ours** |

## Ablations and sensitivity

```bash
python scripts/run_ablation.py --phase views    --datasets Beauty ML-1M SyntheticHard
python scripts/run_ablation.py --phase probes   --datasets Beauty
python scripts/run_ablation.py --phase window   --datasets SyntheticHard
python scripts/make_tables.py --sub ablation
```

*views* is a ladder, not a list of independent removals — each row adds one perspective:

| Row | Configuration | Perspective added |
| --- | --- | --- |
| `L0_longterm` | `--read_win 0` | long-term only (Stage 1 alone) |
| `L1_plus_recent` | `--read_win 8 --read_probes 1` | + recent behaviour (one probe) |
| `L2_full` | `--read_win 8 --read_probes 8` | + multiple interests (= MIRR) |
| `L2p_no_cond` | `… --read_cond 0` | probes no longer conditioned on `u_t` |
| `no_route_bias` | `… --route_bias 0` | drop the long-term interest bias |

*probes* varies `R` with `m` fixed; *window* varies `m` with `R` fixed. The two knobs are
independent: `R` bounds how many interests can be separated at one prediction, `m` is how
much recent behaviour is re-read.

## Slot-collapse diagnostics (appendix)

```bash
python scripts/run_main.py --datasets Beauty --models MIRR MIRR-W ComiRec REMI DT3OR \
       --seeds 2024 --diagnose
# or, on an existing checkpoint:
python scripts/run_diagnose.py --dataset Beauty --model MIRR-W --seed 2024 \
       --ckpt results/checkpoints/Beauty/MIRR-W_s2024.pt
```

`src/analysis/slot_collapse.py` reports `R_norm` (redundancy normalised by a cross-user
baseline), `n_eff` (effective number of interests, comparable across different `R`),
`jaccard_top50` (functional overlap), the load share per slot, and the
**mean-substitution test** — replacing every slot by their mean and re-measuring. If
accuracy does not move, the slots were redundant.

The contrast is the point of the appendix: on the sparse long-tail catalogues
(Beauty, Sports, Toys) a write-time router (`MIRR-W`, `ComiRec`, `DT3OR`) reaches
`R_norm → 1` and `n_eff → 1` — the K slots are copies of one another, and the
mean-substitution test confirms they are *functionally* redundant (accuracy does
not move when they are replaced by their mean).

The effect is dataset-dependent, and the README states this rather than
over-claiming. On ML-1M and SyntheticHard, ComiRec and REMI keep genuine slot
diversity (`n_eff` 1.8–3.0, `R_norm` far below 1); MIRR-W still collapses on all
five datasets (`R_norm` 0.93–1.00). MIRR *never* collapses. Because MIRR also
wins on ML-1M and SyntheticHard, where the baselines do **not** collapse, the
advantage does not depend on the baselines collapsing.

On SyntheticHard the cross-user baseline is not meaningful (the synthetic
catalogue has discrete, shared interest lines, so two different users can own the
same line and their slots are legitimately similar); report raw cosine and
`n_eff` there and ignore `R_norm`.

## Hyper-parameters

Everything is in `src/recipes.py`; nothing is hard-coded in the training script.

| Setting | Value |
| --- | --- |
| Embedding size / layers / heads | 64 / 2 / 2 |
| Max sequence length | 50 (60 on SyntheticHard) |
| Dropout, optimizer, LR | 0.2, Adam, 1e-3 (constant), weight decay 0 |
| Epochs / early-stopping patience | 60 / 8 (100 / 20 on SyntheticHard) |
| Selection criterion | validation NDCG@10 |
| Batching | token-balanced, shared by every method |
| Seeds | 2024, 2025, 2026 |
| MIRR | `R = 8` probes, window `m = 8` **on every dataset**, `M` auto, `α` per-user on Sports/Toys |
| ComiRec / REMI | `K = 4`, 8 sampled supervision positions per sequence |
| TTT baselines | test-time LR searched over {1e-5, 1e-4, 1e-3} on validation |

On catalogs with more than 40k items the batch size is reduced automatically
(128, and 32 for ComiRec) to bound the `(B, T, N)` logits.

One hyper-parameter set is used for all five datasets; nothing is tuned per
dataset except the fusion variant `α`. In particular the read window is `m = 8`
everywhere, including SyntheticHard. A wider window is better on that dataset
(`m = 16` gives R@10 92.9 vs 91.0) — it is reported in the *window* sweep
(`python scripts/run_ablation.py --phase window --datasets SyntheticHard`) rather
than adopted in the main table, precisely so that no dataset needs a special-case
justification.

Adding `--eval_ttt` to a run also executes the streaming test-time-training protocol and
records `ttt_*` metrics plus the latency overhead.

## Citation

```bibtex
@inproceedings{mirr2027,
  title     = {MIRR: Multi-Interest Read-Time Routing for Sequential Recommendation},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM Web Conference (WWW)},
  year      = {2027}
}
```

## License

MIT — see `LICENSE`.
