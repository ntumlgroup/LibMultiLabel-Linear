#!/bin/bash
_train () {
    for d in "${dataset[@]}"; do
        python main.py --linear --seed "$seed" --data_name "$d" \
                        --data_format "$dataformat" \
                        --training_file data/$d/train.$dataformat \
                        --test_file data/$d/test.$dataformat \
                        --liblinear_options "$options" \
                        --linear_technique "$lineartechnique" \
                        --monitor_metrics P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100 \
                        --metric_threshold 0 \
                        --eval_batch_size 256 \
                        --result_dir "$ROOT/$d/$reg/lml" 
    done
}

# --result_dir "$ROOT/$d/$reg/$dirsetting" 
_threshold () {
    for d in "${dataset[@]}"; do
        python threshold_experiment.py --threshold_options "$thresholdoptions" \
                                        --model_path "$ROOT/$d/$reg" \
                                        --result_path "$ROOT/$d/$reg/$dirsetting" \
                                        --dataset_name "$d" \
                                        --linear_technique "$lineartechnique" \
                                        --threshold_technique "$thresholdtechnique"
    done
}

_prediction () {
    for d in "${dataset[@]}"; do
        for filename in "$ROOT/$d/$reg/$dirsetting/models/"*;
        do  
            start=$(date +%s)      
            python main.py --linear --seed "$seed" --data_name "$d" --eval \
                            --data_format "$dataformat" \
                            --training_file data/$d/train.$dataformat \
                            --test_file data/$d/test.$dataformat \
                            --liblinear_options "$options" \
                            --linear_technique "$lineartechnique" \
                            --monitor_metrics P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100 \
                            --metric_threshold 0 \
                            --eval_batch_size 256 \
                            --result_dir "$ROOT/$d/$reg/$dirsetting/logs" \
                            --checkpoint_path "$filename" 
            end=$(date +%s)

            name="$(echo $filename | rev | cut -d '/' -f 1 | rev)" 
            name=${name:0:-7} 
            mv "$ROOT/$d/$reg/$dirsetting/logs/logs.json" "$ROOT/$d/$reg/$dirsetting/logs/$name.json"
            # mv "$ROOT/$d/logs.json" "$ROOT/$d/$name.json"

            echo "${name} | $(($end-$start)) seconds" >> "$ROOT/$d/prediction-time-logs.txt"
        done
    done

}

_generate_graph () {
    # Parse Logs
    for d in "${dataset[@]}"; do
        python experiment_utils.py --log_path "$ROOT/$d/$reg/$dirsetting" \
                                --function_name "create_graph" \
                                --metrics "$graph_metrics" \
                                --dataset_name "$d"
    done
}

eval_metrics="P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100"
graph_metrics="P@1/P@15/NDCG@10/NDCG@5/R@100"
graph_metrics="P@1"

dataset=(
    # "lexglue/ecthr_a"
    # "eur-lex"
    "amazoncat-13k/ver1"
    "rcv1"
    "wiki10-31k"
)

options="-s 1 -e 0.0001 -c 1"
lineartechnique="tree"
dataformat="svm"
thresholdtechnique="thresh_fixed"
seed="1337"
reg="L2"

# geometric series format: geo/a/r/k
# linear series format: lin/start/stop/step
# single thresh format: quantile%
thresholdoptions="geo/.15/.8/100"
ROOT="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/tree"
dirsetting="85-100-geometric-fixed"
cd ..

_train
# _threshold
# _prediction
# _generate_graph

# printf "$ROOT\n$dirsetting\n$options$lineartechnique\n$thresholdoptions $dataformat\n$seed\n$reg" >> "$ROOT/$d/$reg/$dirsetting/run_settings.txt"