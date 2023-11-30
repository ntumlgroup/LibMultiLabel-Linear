
declare -A LEXGLUE_SRCS=( ["SCOTUS"]="scotus" ["ECtHRA"]="ecthr_a" ["ECtHRB"]="ecthr_b" ["EUR-Lex-LexGlue"]="eurlex" ["Unfair-TOS"]="unfair_tos" ["LEDGAR"]="ledgar")
pwd="$(pwd)"

for dataset in "${!LEXGLUE_SRCS[@]}"
do
    # download data
    data_dir="data/$dataset"
    if [ -d ${data_dir} ]; then
        echo "${data_dir} exists, skipped."
    else
        mkdir -p ${data_dir}
        cd ${data_dir}

        if [ $dataset == "SCOTUS" ] || [ $dataset == "LEDGAR" ]; then
            type="multiclass"
        else
            type="multilabel"
        fi
        wget -c https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/$type/${LEXGLUE_SRCS[$dataset]}_lexglue_raw_texts_train.txt.bz2 -O train.txt.bz2
        wget -c https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/$type/${LEXGLUE_SRCS[$dataset]}_lexglue_raw_texts_val.txt.bz2 -O valid.txt.bz2
        wget -c https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/$type/${LEXGLUE_SRCS[$dataset]}_lexglue_raw_texts_test.txt.bz2 -O test.txt.bz2
        bzip2 -d *.bz2
        cd $pwd
    fi
done
