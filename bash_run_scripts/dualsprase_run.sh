#! /bin/bash
cd ..
ROOT="/home/justinchanchan8/tree-prunning-results/runs/primal-dual/model"

dataset=(
    "eur-lex"
    "rcv1"
    "lexglue/ecthr_a"
    "lexglue/ecthr_b"
    "lexglue/ledgar"
    "lexglue/scotus"
    "lexglue/unfair_tos"
)

for d in "${dataset[@]}"; do
    echo "$d" 
    python get_sparsity.py --baseline_path "$ROOT/$d" \
                             --data "$d"

done






