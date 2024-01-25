set -e
for seed in 1109 172;do
    for model in kim_cnn cnn_lwan bigru_lwan;do
        for data in EUR-Lex Wiki10-31K EUR-Lex-57k MIMIC-50;do
            python main.py --config example_config/$data/$model.yml --seed $seed
        done 
    done
done