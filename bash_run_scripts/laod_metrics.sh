#!/bin/bash

ROOT="/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/1vsrest"
thresholdsetting="85-100-quantile"
dataset=(
    # "EUR-Lex"
    # "AmazonCat-13K"
    # "rcv1"
    # "Wiki10-31K"
    # "lexglue/ecthr_a"
    # "lexglue/ecthr_b"
    "lexglue/ledgar"
    # "lexglue/scotus"
    # "lexglue/unfair_tos"
)
logfiles=(
    # "ecthr_a--0.85--0.07846358568764243.pickle.json"    
    # "ecthr_a--0.904--0.12143632562157028.pickle.json"    
    # "ecthr_a--0.95085--0.2082280621544385.pickle.json"
    # "ecthr_a--0.99175--0.5371938667671551.pickle.json"
    # "ecthr_b--0.85--0.08008837505824691.pickle.json"   
    # "ecthr_b--0.904--0.12431434296403653.pickle.json"
    # "ecthr_b--0.95085--0.21129666914132697.pickle.json"
    # "ecthr_b--0.99175--0.5455919182185529.pickle.json"
    "ledgar--0.85--0.3010187809548817.pickle.json"
    "ledgar--0.904--0.40120806979342655.pickle.json"
    "ledgar--0.95085--0.5634711742003374.pickle.json"
    "ledgar--0.99175--1.0397890124185338.pickle.json"
    # "scotus--0.85--0.0377690402182125.pickle.json"
    # "scotus--0.904--0.06311680380751629.pickle.json"
    # "scotus--0.95085--0.11788842427419201.pickle.json"
    # "scotus--0.99175--0.36202061472749536.pickle.json"
    # "unfair_tos--0.85--0.11519069728143766.pickle.json"
    # "unfair_tos--0.904--0.1562556015336804.pickle.json"
    # "unfair_tos--0.95085--0.22359666753697752.pickle.json"
    # "unfair_tos--0.99175--0.49396315654312095.pickle.json"


)

cd ..

for d in "${dataset[@]}"; do
    for file in "${logfiles[@]}"; do
        python experiment_utils.py --log_path "$ROOT/$d/$thresholdsetting/logs/$file" \
                                --function_name "metric_loader" \
                                --metrics "P@1/P@3/P@5/R@3/R@5"
    done
done