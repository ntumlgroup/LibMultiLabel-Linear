#! /bin/bash
cd ..
ROOT="runs/"
dataset=(
    # "w7a"
    # "w3a"
    # "svmguide1"
    # "hyperpartisan-news-detection"
    # "duke-breast-cancer"
    # "a9a"
    # "a4a"
    # "a1a"
    # "news20" 
    # "rcv1"
    # "real-sim"  
    # "EUR-Lex"
    # "Wiki10-31K"
    "AmazonCat-13K"
)


#note val set used for testing: duke, 
for data in "${dataset[@]}"; do
    echo "$data"
    python get_sparsity.py --root_path "$ROOT" \
                    --l1_name "baseline_model_AmazonCat-13K-L1" \
                    --l2_name "baseline_model_AmazonCat-13K" 
done


