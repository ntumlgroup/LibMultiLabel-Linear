#!/bin/bash
lml_prediction () {
    for d in "${dataset[@]}"; do
        for filename in "$ROOT/$d/$reg/lml/"*;
        do  

            python main.py --linear --seed "$seed" --data_name "$d" --eval \
                            --data_format "$dataformat" \
                            --training_file data/$d/train.$dataformat \
                            --test_file data/$d/test.$dataformat \
                            --liblinear_options "$options" \
                            --linear_technique "$lineartechnique" \
                            --monitor_metrics P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100 \
                            --metric_threshold 0 \
                            --eval_batch_size 256 \
                            --result_dir "$ROOT/$d/$reg/lml" \
                            --checkpoint_path "$filename" 
        done
    done

}

full_split_prediction () {
    for d in "${dataset[@]}"; do
        echo "$d full size prediction 80/20 split"
        for filename in "$ROOT/$d/$reg/$dirsetting/"*;
        do  
            if [[ $filename != "$ROOT/$d/$reg/$dirsetting/linear_pipeline.pickle" ]]; then
                continue
            fi
            python experiment_utils.py --function_name "thresh_node_predict" \
                            --dataset_name "$d" \
                            --model_path "$filename" \
                            --log_path "$ROOT/$d/$reg/$dirsetting" \
                            --metrics "P@1/P@2/P@3/P@4/P@5/P@6/P@7/P@8/P@9/P@10/P@11/P@12/P@13/P@14/P@15/P@16/P@17/P@18/P@19/P@20/NDCG@1/NDCG@2/NDCG@3/NDCG@4/NDCG@5/NDCG@6/NDCG@7/NDCG@8/NDCG@9/NDCG@10/NDCG@11/NDCG@12/NDCG@13/NDCG@14/NDCG@15/NDCG@16/NDCG@17/NDCG@18/NDCG@19/NDCG@20/R@5/R@10/R@20/R@50/R@75/R@100" 
        done
    done

}

threshold_split_prediction () {
    root="/home/justinchanchan8/LibMultiLabel2/runs/iter_thresh_global"
    for d in "${dataset[@]}"; do
        echo "$d threshold prediction 80/20 split"
        for filename in "$root/$d/models/"*;
        do  
            python experiment_utils.py --function_name "thresh_node_predict" \
                            --dataset_name "$d" \
                            --model_path "$filename" \
                            --log_path "$root/$d/logs" \
                            --metrics "P@1/P@2/P@3/P@4/P@5/P@6/P@7/P@8/P@9/P@10/P@11/P@12/P@13/P@14/P@15/P@16/P@17/P@18/P@19/P@20/NDCG@1/NDCG@2/NDCG@3/NDCG@4/NDCG@5/NDCG@6/NDCG@7/NDCG@8/NDCG@9/NDCG@10/NDCG@11/NDCG@12/NDCG@13/NDCG@14/NDCG@15/NDCG@16/NDCG@17/NDCG@18/NDCG@19/NDCG@20/R@5/R@10/R@20/R@50/R@75/R@100" 
        done
    done

}


eval_metrics="P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100"
graph_metrics="P@1/P@15/NDCG@10/NDCG@5/R@100"
graph_metrics="P@1"

dataset=(
    # "lexglue/ecthr_a"
    "eur-lex"
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
# thresholdoptions="geo/.15/.8/100"
ROOT="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/tree"
dirsetting="0-100-geometric-node-tr-val-split"
cd ..

lml_prediction
# full_split_prediction
# threshold_split_prediction
# _prediction
# _generate_graph

# printf "$ROOT\n$dirsetting\n$options$lineartechnique\n$thresholdoptions $dataformat\n$seed\n$reg" >> "$ROOT/$d/$reg/$dirsetting/run_settings.txt"