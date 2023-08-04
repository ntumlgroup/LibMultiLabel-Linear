mkdir -p logs
mkdir -p models


for data in "wiki10-31k";
do
  for num_models in 100 50 20 10;
  do
    for beam_width in 10 100000;
    do
      for seed in 1 2 3 4 5 6 7 27 100 9527;
      do
        for K in 100;
        do
          for sample_rate in 0.1 0.2 0.3 0.4 0.5;
          do
            head="Rand-label-Forest-${num_models}"
            param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
            name="${head}_${data}_${param}" 
            
            if [ ! -f ./logs/${name}.log ];
            then
              python3 bagging_linear.py \
    	        --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
    	        --sample_rate ${sample_rate} --datapath "./datasets/libsvm-format/${data}" \
    	        |tee ./logs/${name}.log
            fi
          done
        done
      done
    done
  done
done

for data in "wiki10-31k";
do
  for num_models in 10 5 3 1;
  do
    for beam_width in 10 100000;
    do
      for seed in 1 2 3 4 5 6 7 27 100 9527;
      do
        for K in 100;
        do
          for sample_rate in 1.1;
          do
            head="Rand-label-Forest-${num_models}"
            param="seed=${seed}_K=${K}_beam-width=${beam_width}_sample-rate=${sample_rate}"
            name="${head}_${data}_${param}" 
            
            if [ ! -f ./logs/${name}.log ];
            then
	      echo "run ${name} !!"
              python3 bagging_linear.py \
    	        --num_models ${num_models} --seed ${seed} --K ${K} --beam_width ${beam_width} \
    	        --sample_rate ${sample_rate} --datapath "./datasets/libsvm-format/${data}" \
    	        |tee ./logs/${name}.log
            fi
          done
        done
      done
    done
  done
done
