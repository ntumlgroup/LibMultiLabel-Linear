set -e
for seed in 0 1109 172 309;do
    for model in rnn_no_tanh rnn;do
        for task in MIMIC-50 SCOTUS;do
            python main.py --config example_config/$task/$model.yml --seed $seed
        done
    done
done