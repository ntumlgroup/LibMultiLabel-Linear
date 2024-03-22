set -e
for seed in 0 1109;do
    for model in bilstm_tanhW_mlp_tune bilstm_tanhW_tune cnn_tanhW_mlp_tune cnn_tanhW_tune;do
        python search.py --exps $seed/$model --config example_config/EUR-Lex/tune/$model.yml --template example_config/EUR-Lex/template.yml --seed $seed --no_checkpoint 
    done
done
# 1109 172 309
# bilstm_tanhW_mlp_tune bilstm_tanhW_tune bilstm_tanh_mlp_tune bilstm_tanh_tune bilstm_vanilla_mlp_tune bilstm_vanilla_tune cnn_tanhW_mlp_tune cnn_tanhW_tune cnn_tanh_mlp_tune cnn_tanh_tune cnn_vanilla_mlp_tune cnn_vanilla_tune