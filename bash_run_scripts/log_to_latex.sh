#! /bin/bash
cd ..
ROOT="tree-prunning-results/runs"
dataset=(
    # "w7a"
    # "w3a"
    # "svmguide1"
    # "hyperpartisan-news-detection"
    # "duke-breast-cancer"
    "a9a"
    # "a4a"
    # "a1a"
)

reg=(
    "l1"
    "l2"
)
models=(
    "l1-a9a-model"
    "l2-a9a-model"
    "l1-news20-model"
    "l2-news20-model"
    "l1-real-sim-model"
    "l2-real-sim-model"
)

#note val set used for testing: duke, 
# for data in "${dataset[@]}"; do
#     for r in "${reg[@]}"; do
#         echo "$r"
#         python print_metrics.py --log_path "$ROOT/binary_classification/$data/$r-0-$data-log.json" \
#                         --model_path "$ROOT/binary_classification/$data/model/${r}-0-pipeline.pickle"
#     done
# done

for m in "${models[@]}"; do
    python print_metrics.py --model_path $m \
    
    done


