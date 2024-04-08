set -e
data=$1
for seed in 0 1109 172 309;do
    for model in bilstm_tanhW_mlp_tune bilstm_tanhW_tune bilstm_tanh_mlp_tune bilstm_tanh_tune bilstm_vanilla_mlp_tune bilstm_vanilla_tune cnn_tanhW_mlp_tune cnn_tanhW_tune cnn_tanh_mlp_tune cnn_tanh_tune cnn_vanilla_mlp_tune cnn_vanilla_tune;do
        python search_params.py --config example_config/$data/tune/$model.yml --no_retrain --no_checkpoint --seed $seed
    done
done
