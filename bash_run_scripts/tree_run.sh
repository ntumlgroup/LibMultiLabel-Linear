#!/bin/bash
reg=(
    "l2svm_l2r"
)

dataset=(
    # "EUR-Lex"
    # "AmazonCat-13K/ver1"
    # "rcv1" -> which rcv1?
    # "Wiki10-31K"
)

# geometric series format: geo/a/r/k
# linear series format: lin/start/stop/step
# single thresh format: quantile%
thresholdoptions="geo/.15/.8/100"
ROOT="/home/justinchanchan8/tree-prunning-results/runs"
cd ..


# run thresholding
# for d in "${dataset[@]}"; do
#     python threshold_experiment.py --threshold_options "$thresholdoptions" \
#                                     --model_path "$ROOT/$d/L2-baseline" \
#                                     --result_path "$ROOT/$d/L2-baseline-V2/85-100-quantile-models" \
#                                     --dataset_name "$d" \
#                                     --linear_technique "tree"
# done



# Prediction
# for d in "${dataset[@]}"; do
#     for filename in "$ROOT/$d/L2-baseline-V2/85-100-quantile-models"/*;
#     do
#         start=$(date +%s)      
#         name="$(echo $filename | cut -d '/' -f 9)"
#         python main.py --eval --linear \
#                         --config "example_config/$d/tree_l2svm.yml" \
#                         --checkpoint_path "$filename" \
#                         --log_name "${d}-${name}" \
#                         --result_dir "$ROOT/$d/L2-baseline-V2/85-100-quantile-logs" 
#         end=$(date +%s)
#         echo "${d}-${name} | $(($end-$start)) seconds" >> "$ROOT/$d/L2-baseline-V2/85-100-quantile-logs/test-logs.txt"
#     done
# done


# for filename in "$ROOT/Wiki10-31K/L2-baseline/85-100-quantile"/*;
# do
#     start=$(date +%s)      
#     name="$(echo $filename | cut -d '/' -f 9)"
#     python main.py --eval --linear \
#                     --config "example_config/Wiki10-31K/tree_l2svm.yml" \
#                     --checkpoint_path "$filename" \
#                     --log_name "Wiki10-31K-${name}" \
#                     --result_dir "$ROOT/Wiki10-31K/L2-baseline/85-100-quantile-logs" 
#     end=$(date +%s)
#     echo "${d}-${name} | $(($end-$start)) seconds" >> "$ROOT/Wiki10-31K/L2-baseline/85-100-quantile-logs/test-logs.txt"
# done

# Parse Logs
python experiment_utils.py --log_path "$ROOT/EUR-Lex/L2-baseline-V4.1/85-100-quantile-logs" \
                           --function_name "create_graph" \
                           --metrics "P@1/P@15/NDCG@10/NDCG@5/R@100"

# baseline 5,659,164,958
# threshold@99 5.659.165,066
