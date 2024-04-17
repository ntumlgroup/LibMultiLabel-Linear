for exp in runs/*;do
    echo $exp
    best=$(dirname $(python find_best.py -r $exp))
    root=$(dirname $best)
    cp $best/events* $root/trial_best_params
done