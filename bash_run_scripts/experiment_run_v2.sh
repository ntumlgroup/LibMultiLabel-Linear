# for filename in runs/baseline_model_Wiki10-31K/Experiment/*; 
# do 
#     start=$(date +%s)
#     echo $filename
#     python main.py --config example_config/Wiki10-31K/tree_l2svm.yml \
#                     --test_file data/Wiki10-31K/test.txt \
#                     --eval \
#                     --linear \
#                     --data_format txt \
#                     --checkpoint_path $filename \
#                     --model_folder_name $filename
#     end=$(date +%s)
#     echo "Elapsed Time: $(($end-$start)) seconds"
    
#  done
cd ..

python main.py --config example_config/EUR-Lex/tree_l2svm.yml \
                --test_file data/EUR-Lex/test.txt \
                --eval \
                --linear \
                --data_format txt \
                --checkpoint_path "/home/justinchanchan8/tree-prunning-results/runs/EUR-Lex/L2-baseline/linear_pipeline.pickle" \
                --result_dir tree-prunning-results/runs/EUR-Lex/L2-baseline \
                --model_folder_name 99-results

# python main.py --config example_config/AmazonCat-13K/tree_l2svm.yml \
#                 --test_file data/AmazonCat-13K/test.txt \
#                 --eval \
#                 --linear \
#                 --data_format txt \
#                 --checkpoint_path tree-prunning-results/runs/AmazonCat-13K/L2-baseline-V2/AmazonCat-13K-85-90-95/0.9-0.16561050358681018.pickle \
#                 --result_dir tree-prunning-results/runs/AmazonCat-13K/L2-baseline-V2 \
#                 --model_folder_name 90-results

# python main.py --config example_config/AmazonCat-13K/tree_l2svm.yml \
#                 --test_file data/AmazonCat-13K/test.txt \
#                 --eval \
#                 --linear \
#                 --data_format txt \
#                 --checkpoint_path tree-prunning-results/runs/AmazonCat-13K/L2-baseline-V2/AmazonCat-13K-85-90-95/0.95-0.28167001603906777.pickle \
#                 --result_dir tree-prunning-results/runs/AmazonCat-13K/L2-baseline-V2 \
#                 --model_folder_name 95-results


# python main.py --config example_config/AmazonCat-13K/tree_l2svm.yml \
#                 --test_file data/AmazonCat-13K/test.txt \
#                 --eval \
#                 --linear \
#                 --data_format txt \
#                 --checkpoint_path tree-prunning-results/runs/AmazonCat-13K/L2-baseline/Experiment/0.4737682958024462.pickle \
#                 --result_dir tree-prunning-results/runs/AmazonCat-13K/L2-baseline-V2 \
#                 --model_folder_name 99-results