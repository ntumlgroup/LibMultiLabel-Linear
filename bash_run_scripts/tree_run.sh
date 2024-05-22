#!/bin/bash
reg="L2"


dataset=(
    "eur-lex"
    # "AmazonCat-13K/ver1"
    # "rcv1" -> which rcv1?
    # "Wiki10-31K"
)
dataformat="txt"

# geometric series format: geo/a/r/k
# linear series format: lin/start/stop/step
# single thresh format: quantile%

thresholdoptions="geo/.15/.8/100"
ROOT="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/tree"
dirsetting="85-100-geometric"
cd ..


# training
for d in "${dataset[@]}"; do
    python main.py --linear \
                    --data_format "$dataformat" \
                    --training_file data/$d/train.$dataformat \
                    --test_file data/$d/test.$dataformat \
                    --liblinear_options "-s 1 -e 0.0001 -c 1" \
                    --linear_technique "tree" \
                    --monitor_metrics P@1 P@3 P@5 \
                    --metric_threshold 0 \
                    --eval_batch_size 256 
done


# run thresholding
# for d in "${dataset[@]}"; do
#     python threshold_experiment.py --threshold_options "$thresholdoptions" \
#                                     --model_path "$ROOT/$d/$reg" \
#                                     --result_path "$ROOT/$d/$reg/$dirsetting" \
#                                     --dataset_name "$d" \
#                                     --linear_technique "tree"
# done



# # Prediction
# for d in "${dataset[@]}"; do
#     for filename in "$ROOT/$d/$reg/$dirsetting/models/"*;
#     do  
#         start=$(date +%s)      
#         name="$(echo $filename | rev | cut -d '/' -f 1 | rev)" 
        
#         python main.py --linear --eval \
#                         --data_format "$dataformat" \
#                         --training_file data/$d/train.$dataformat \
#                         --test_file data/$d/test.$dataformat \
#                         --liblinear_options "-s 1 -e 0.0001 -c 1" \
#                         --linear_technique "tree" \
#                         --monitor_metrics P@1 P@2 P@3 P@4 P@5 P@6 P@7 P@8 P@9 P@10 P@11 P@12 P@13 P@14 P@15 P@16 P@17 P@18 P@19 P@20 NDCG@1 NDCG@2 NDCG@3 NDCG@4 NDCG@5 NDCG@6 NDCG@7 NDCG@8 NDCG@9 NDCG@10 NDCG@11 NDCG@12 NDCG@13 NDCG@14 NDCG@15 NDCG@16 NDCG@17 NDCG@18 NDCG@19 NDCG@20 R@5 R@10 R@20 R@50 R@75 R@100 \
#                         --metric_threshold 0 \
#                         --eval_batch_size 256 \
#                         --checkpoint_path "$filename" \
#                         --result_dir "$ROOT/$d/$reg/$dirsetting/logs" 
#         end=$(date +%s)
#         name=${name:0:-7} 
#         mv "$ROOT/$d/$reg/$dirsetting/logs/logs.json" "$ROOT/$d/$reg/$dirsetting/logs/$name.json"

#         echo "${name} | $(($end-$start)) seconds" >> "$ROOT/$d/$reg/$dirsetting/test-logs.txt"
#     done
# done




# # Parse Logs
# for d in "${dataset[@]}"; do
#     python experiment_utils.py --log_path "$ROOT/$d/$reg/$dirsetting" \
#                             --function_name "create_graph" \
#                             --metrics "P@1/P@15/NDCG@10/NDCG@5/R@100" \
#                             --dataset_name "$d"
# done
# baseline 5,659,164,958
# threshold@99 5.659.165,066