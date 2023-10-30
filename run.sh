config=$1
ccs_id=$2

python search_params.py --config "$config" --no_retrain
twccli rm ccs -f -s "$ccs_id"