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
    "news20" 
    "rcv1"
    "real-sim"  
)

# options=(
#     "-s 7 -c .25 -e 0.01 -q"
#     "-s 7 -c  1 -e 0.01 -q"
#     "-s 7 -c  4 -e 0.01 -q"
#     "-s 7 -c 16 -e 0.01 -q"
#     # "-s 6 -c .25 -e 0.1 -q"
#     # "-s 6 -c 1 -e 0.1 -q"
#     # "-s 6 -c 4 -e 0.1 -q"
#     # "-s 6 -c 16 -e 0.1 -q"
#     # "-s 6 -c .25 -e 1 -q"
#     # "-s 6 -c 1 -e 1 -q"
#     # "-s 6 -c 4 -e 1 -q"
#     # "-s 6 -c 16 -e 1 -q"
#     # "-s 6 -c .25 -e .001 -q"
#     # "-s 6 -c 1 -e .001 -q"
#     # "-s 6 -c 4 -e .001 -q"
#     # "-s 6 -c 16 -e .001 -q"
#     # ""
#     "-e .001 -c 4" "-s"

# )

#note val set used for testing: duke, 
for data in "${dataset[@]}"; do
    echo "$data"
    #l2
    python binary_run.py --data_path "data/binary_dataset/$data/train.txt" \
                    --output_path "$ROOT/binary_classification/models" \
                    --model_name "l2-$data-model" \
                    --liblinear_options "-s 0 -c 4 -e 0.001 -q" \
    #l1
    python binary_run.py --data_path "data/binary_dataset/$data/train.txt" \
                    --output_path "$ROOT/binary_classification/models" \
                    --model_name "l1-$data-model" \
                    --liblinear_options "-s 6 -c 4 -e 0.001 -q" \

done


