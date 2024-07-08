#!/bin/bash
reg="L2"

dataset=(
    "rcv1"
    "eur-lex"
    "mimic"
    "wiki10-31k"
    "amazoncat-13k/ver1"
)
dataformat="txt"

ROOT="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/tree"
cd ..


# training
# for d in "${dataset[@]}"; do
#     echo "$d run"
#     if [[ "$d" == "amazoncat-13k/ver1" ]]; then
#         dataformat="svm"
#     fi
#     python main.py --linear \
#                     --data_format "$dataformat" \
#                     --data_name "$d" \
#                     --training_file data/$d/train.$dataformat \
#                     --test_file data/$d/test.$dataformat \
#                     --result_dir "runs/lml/$d" \
#                     --liblinear_options "-s 1 -e 0.0001 -c 1" \
#                     --linear_technique "tree" \
#                     --monitor_metrics P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100 \
#                     --metric_threshold 0 \
#                     --eval_batch_size 256 
# done


python test_iter_thresh.py