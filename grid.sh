mkdir -p logs
mkdir -p models


# for data in "wiki10-31k";
# do
#   for num_models in 10 5 3 1;
#   do
#     for beam_width in 10 100000;
#     do
#       for seed in 1 2 3 4 5 6 7 27 100 9527;
#       do
#         for K in 100;
#         do
#           for sample_rate in 1.1;
#           do
#             head="Rand-label-Forest-${num_models}"
#             param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
#             name="${head}_${data}_${param}" 
#             
#             if [ ! -f ./logs/${name}.log ];
#             then
# 	      echo "run ${name} !!"
#               python3 bagging_linear.py \
#     	        --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
#     	        --sample_rate ${sample_rate} --datapath "./datasets/libsvm-format/${data}" \
#     	        |tee ./logs/${name}.log
#             fi
#           done
#         done
#       done
#     done
#   done
# done

#for data in "amazon-670k-pecos";
for data in "amazon-3m-pecos";
#for data in "eur-lex" "wiki10-31k" "amazoncat-13k" "amazon-670k-pecos";
do
  #for num_models in 100 50;
  for num_models in 100;
  #for num_models in 1;
  do
    for beam_width in 10;
    do
      #for seed in 1 2 3 4 5 6 7 27 100 9527;
      for seed in 1;
      do
        for K in 100;
        do
          for sample_rate in 0.1;
          #for sample_rate in 1.1;
          do
            head="Rand-label-Forest-${num_models}"
            param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
            name="${head}_${data}_${param}" 
            
            if [ ! -f ./logs/${name}.log ];
            then
              export MKL_INTERFACE_LAYER=ILP64
              python3 bagging_linear.py \
    	        --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
    	        --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
    	        |tee ./logs/${name}.log
            fi
          done
        done
      done
    done
  done
done




# #for data in "amazon-670k-pecos";
# for data in "amazon-3m-pecos";
# #for data in "eur-lex" "wiki10-31k" "amazoncat-13k";
# do
#   #for num_models in 100 50;
#   for num_models in 100;
#   #for num_models in 1000;
#   do
#     #for beam_width in 10 100000;
#     for beam_width in 10;
#     do
#       #for seed in 1 2 3 4 5 6 7 27 100 9527;
#       for seed in 1;
#       do
#         for K in 100;
#         do
#           #for sample_rate in 0.01;
#           for sample_rate in 0.1;
#           do
#             for model_idx in {0..99};
#             do
#               head="Rand-label-Forest-${num_models}"
#               param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
#               name="${head}_${data}_${param}" 
#               
#               # python3 bagging_linear.py \
#     	      #   --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} --idx ${model_idx} \
#     	      #   --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
#     	      #   |tee ./logs/${name}.log
#               
#               if [ ! -f ./logs/${name}.log ];
#               then
#                 export MKL_INTERFACE_LAYER=ILP64
#                 python3 bagging_linear.py \
#     	          --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} --idx ${model_idx} \
#     	          --sample_rate ${sample_rate} --datapath "./datasets/pickle-format/${data}" \
#     	          |tee ./logs/${name}-${model_idx}.log
#               fi
#             done
#           done
#         done
#       done
#     done
#   done
# done


