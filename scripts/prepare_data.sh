#!/usr/bin/env bash
# Prepare every dataset used in the paper.
#
# 1. Download the raw files listed in data/raw/README.md and place them in data/raw/.
# 2. Run this script from the repository root:
#       bash scripts/prepare_data.sh
#
# It writes data/processed/<name>/{data.pkl, meta.json} for the four Amazon
# benchmarks and ML-1M, and generates the synthetic benchmark. No dataset is
# committed to the repository.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== preprocessing public benchmarks =="
python -m src.data.preprocess --datasets Beauty Sports Toys CDs ML-1M

echo "== generating the synthetic multi-interest benchmark =="
python -m src.data.synthetic --preset hard

echo "== done; processed datasets are in data/processed/ =="
