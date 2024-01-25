#!/bin/bash
ROOT="/home/justinchanchan8/tree-prunning-results/runs/1vsrest"
cd ..


thresholdsetting="0-100-quantile"

reg=(
    "l2svm_l2r"
)

dataset=(
    # "EUR-Lex"
    # "AmazonCat-13K"
    # "rcv1" -> which rcv1?
    # "Wiki10-31K"
    "lexglue/ecthr_a"
    # "lexglue/ecthr_b"
    "lexglue/ledgar"
    "lexglue/scotus"
    "lexglue/unfair_tos"
)

# thresholdoptions="geo/.15/.8/100"
thresholdoptions="lin/0/100"

# Training
# python main.py --config example_config/lexglue/ecthr_a/1vsrest_l2svm.yml \
#                --training_file data/lexglue/ecthr_a/train_val.txt \
#                --test_file data/lexglue/ecthr_a/test.txt \
#                --linear \
#                --result_dir "$ROOT/lexglue/ecthr_a/0-100-quantile/models" \
#                --linear_technique "1vsrest" \
#                --data_format txt


echo "Threshold Experiment"
for d in "${dataset[@]}"; do
    python threshold_experiment.py --threshold_options "$thresholdoptions" \
                                    --model_path "$ROOT/$d/l2svm_l2r" \
                                    --result_path "$ROOT/$d/$thresholdsetting" \
                                    --dataset_name "$d" \
                                    --linear_technique "1vsrest"
done


echo "Prediction"
for d in "${dataset[@]}"; do
    for filename in "$ROOT/$d/$thresholdsetting/models/"*;
    do
        echo $filename
        start=$(date +%s)      
        name="$(echo $filename | rev | cut -d '/' -f -1 | rev)"

        python main.py --linear --eval \
                        --data_format txt d\
                        --training_file data/$d/train_val.txt \
                        --test_file data/$d/test.txt \
                        --liblinear_options="-s 1 -e 0.0001 -c 1" \
                        --monitor_metrics P@1 P@3 P@5 R@3 R@5 NDCG@3 NDCG@5 \
                        --checkpoint_path "$filename" \
                        --log_name "${name}" \
                        --result_dir "$ROOT/$d/$thresholdsetting/logs" 

        end=$(date +%s)
        echo "${d}-${name} | $(($end-$start)) seconds" >> "$ROOT/$d/$thresholdsetting/logs/test-logs.txt"
    done
done



# echo "Graph"
# for d in "${dataset[@]}"; do
#     python experiment_utils.py --log_path "$ROOT/$d/$thresholdsetting/logs" \
#                             --function_name "create_graph" \
#                             --metrics "P@1/P@15/NDCG@10/NDCG@5/R@100"
#                             # --metrics "P@1/R@5"
# done