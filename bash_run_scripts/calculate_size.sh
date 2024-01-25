#! /bin/bash

data="EUR-Lex"
ROOT="tree-prunning-results/runs"
baseline_dir="L2-baseline-V4"
experiment_dir="85-100-quantile"
models=(
    "linear_pipeline.pickle"
    "0--0.85--0.011785717207408493.pickle"
    "9--0.90546--0.02382156999333146.pickle"
    "22--0.95147--0.05251682562911752.pickle"
    "83--0.99788--0.46611385727376153.pickle"
)

for m in "${models[@]}"; do
    python check.py --model_path "$ROOT/$data/$baseline_dir/$experiment_dir/$m"
done

data="EUR-Lex"
ROOT="tree-prunning-results/runs"
baseline_dir="L1-baseline"
experiment_dir="85-100-quantile"
models=(
    "linear_pipeline.pickle"
    # "40--0.98072--0.12090685679545396.pickle"
    # "53--0.9901--0.19687946847033738.pickle"
)

for m in "${models[@]}"; do
    python check.py --model_path "$ROOT/$data/$baseline_dir/$m"
done
