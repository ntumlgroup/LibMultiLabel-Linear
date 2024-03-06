set -e
root="runs"
metric=$1
dirs=$(ls $root)
for dir in $dirs;do
    files=$(find $(pwd)/$root/$dir -type f -name "error.txt" -exec dirname {} \; | sort -u)
    if [[ ! -z "$files" ]]; then
        for i in $files;do
            # echo $i
            python main.py --config $i/params.json --result_dir $root/$dir
        done
        best=$(python find_best.py -r $root/$dir -m $metric)
        echo $best
        cp -r $(dirname $best) $root/$dir/trial_best_params
        mv $root/$dir/trial_best_params/params.json $root/$dir/trial_best_params/results.json
        python main.py --config $(dirname $best)/params.json --result_dir $root/$dir --custom_run_name trial_best_params
    fi
done
