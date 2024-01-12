for model in kim_cnn cnn_lwan bigru_lwan;do
    for data in EUR-LEX Wiki10-31K EUR-LEX-57k ;do
        python main.py --config example_config/$data/$model.yml
    done 
done