config=$1
ccs_id=$2

while IFS= read -r line
do
    python search_params.py --config "$line" --no_retrain 
done < "$config"
twccli rm ccs -f -s "$ccs_id"