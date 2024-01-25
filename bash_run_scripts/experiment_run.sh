#!/bin/bash
reg=(
    "l2svm_l2r"
)

dataset=(
    # "EUR-Lex"
    # "AmazonCat-13K/ver1"
    # "rcv1" -> which rcv1?
    # "Wiki10-31K"
    "lexglue/ecthr_a"
    "lexglue/ecthr_b"
    "lexglue/ledgar"
    "lexglue/scotus"
    "lexglue/unfair_tos"
)

# geometric series format: geo/a/r/k
# linear series format: lin/start/stop/step
# single thresh format: quantile%
thresholdoptions=".99"
ROOT="/home/justinchanchan8/tree-prunning-results/runs/1vsrest"
cd ..


# run thresholding
# for d in "${dataset[@]}"; do
#     for r in "${reg[@]}"; do
#         python threshold_experiment.py --threshold_options "$thresholdoptions" \
#                                         --model_path "$ROOT/1vsrest/$d/$r" \
#                                         --result_path "$ROOT/1vsrest/99_models/$d" \
#                                         --dataset_name "$d" \
#                                         --linear_technique "1vsrest"
#     done
# done



#l2
# for l in "${lexglue[@]}"; do
#     for filename in "$ROOT/1vsrest/99_models/lexglue/$l"/*;
#     do
#         echo "filename path: ${filename}"
#         python get_sparsity.py --baseline_path "$ROOT/1vsrest/lexglue/$l/l2svm_l2r" \
#                 --threshold_path "$filename" 
#     done
# done

#l1
# for l in "${lexglue[@]}"; do
#     python get_sparsity.py --baseline_path "$ROOT/1vsrest/lexglue/$l/l2svm_l1r" 
# done


# Prediction
# for d in "${dataset[@]}"; do
#     filename="$ROOT/99_models/$d"

#     start=$(date +%s)      
#     python main.py --eval --linear \
#                     --config "example_config/$d/1vsrest_l2svm.yml" \
#                     --checkpoint_path "$filename"/* \
#                     --log_name "${d}-${thresholdoptions}" \
#                     --liblinear_options "-s 1 -e 0.0001 -c 1 -q" \
#                     --result_dir "$ROOT/99_models/pred_logs" 
#     end=$(date +%s)
#     echo "${d}-${thresholdoptions} | -s 1 -e 0.0001 -c 1 -q | $(($end-$start)) seconds" >> "$ROOT/99_models/pred_logs/test_logs.txt"
# done

# Parse Logs
# for filename in "$ROOT/99_models/pred_logs/lexglue"/*;
# do
#     if [[ "$filename" == *".json"* ]]; then
#         echo "$(echo $filename | cut -d '/' -f 10)"
#         python parse_experiment.py --log_path "$filename" 

#     fi
# done

for filename in "$ROOT/99_models/pred_logs"/*;
do
    if [[ "$filename" == *".json"* ]]; then
        echo "$(echo $filename | cut -d '/' -f 10)"
        python parse_experiment.py --log_path "$filename" \
                            --function_name "metric_loader"

    fi
done


# baseline 5,659,164,958
# threshold@99 5.659.165,066
