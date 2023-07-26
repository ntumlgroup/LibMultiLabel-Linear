#!/bin/bash

# basic seting
config=$1
epochs=5
#m=0.9
#wd=0.0

task(){
# Set up train command
train_cmd="python3 main.py"
train_cmd="${train_cmd} --config ${config}"
train_cmd="${train_cmd} --epochs ${epochs}"
#train_cmd="${train_cmd} --momentum ${m}"
#train_cmd="${train_cmd} --weight_decay ${wd}"

# Print out all parameter pair
for solver in 'adam' 'sgd'
do
    for lr in `seq -4 3` # real_lr = 10^{lr}
    do
        for bs in `seq 2 2` # real_bs = 4^{bs}
        do
            rlr=`echo "scale = 6; 10^${lr}" | bc`
            rbs=`echo "scale = 14; 4^${bs}" | bc`
            cmd="${train_cmd} --optimizer $solver"
            cmd="${cmd} --learning_rate ${rlr}"
            cmd="${cmd} --batch_size ${rbs}"
            echo "${cmd}"
        done
    done
done
}

# Check command
task
wait

# Run
#task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
