for exp in runs/*
    best=$(dirname $(python find_best.py -r $exp)
    root=$(dirname $best)
    cp $best/events* $root/trial_best_params