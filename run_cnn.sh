for seed in 0 1109 172 309;do
    for model in cnn_tune_no_tanh ;do
        for task in MIMIC-50 SCOTUS;do
            python search_params.py --config example_config/$task/$model.yml --seed $seed --no_checkpoint --no_retrain
        done
    done
done
