import libmultilabel.linear as linear
import time
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--beam_width', type=int, default=10000)
parser.add_argument('--num_models', type=int, default=3)
parser.add_argument('--sample_rate', type=float, default=0.1)
parser.add_argument('--datapath', type=str, default="")
parser.add_argument('--idx', type=int, default=-1)

ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

print("start", flush=True)
start = time.time()
with open(ARGS.datapath + '.pkl', "rb") as F:
    datasets = pickle.load(F)
print("data loading cost:", time.time()-start, flush=True)

training_start = time.time()

num_models = ARGS.num_models

seed_pool = []
while len(seed_pool) != num_models:
    seed = np.random.randint(2**31 - 1)
    if seed not in seed_pool:
        seed_pool += [seed]

model_name = "Rand-label-partitions-No-replacement_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
#model_name = "Rand-label-Forest-No-replacement_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )
# model_name = "OVR_{data}_seed={seed}_machine-idx={idx}.model".format(
#         seed = ARGS.seed,
#         data = os.path.basename(ARGS.datapath),
#         idx = ARGS.idx
#         )

if ARGS.idx >= 0:
    model_idx = ARGS.idx
    np.random.seed(seed_pool[model_idx])

    model_start = time.time()
    submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    if not os.path.isfile(submodel_name):
        #model, metalabels = linear.train_tree_partition(
        model, metalabels = linear.train_tree(
            datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q",K=ARGS.K)
        #level_0_model, level_1_model, indices = linear.train_tree_subsample(
        #        datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -m 32 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
        print("training one model cost:", time.time()-model_start, flush=True)
    # submodel_name = "./models/" + model_name
    # tmp, indices = linear.train_1vsrest_distributed(
    #         datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -m 32 -q", machine_idx=ARGS.idx)
    with open(submodel_name, "wb") as F:
        #pickle.dump((level_0_model, level_1_model, indices), F, protocol=5)
        pickle.dump((model, metalabels), F, protocol=5)

else:
    for model_idx in range(num_models):
        submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    
        np.random.seed(seed_pool[model_idx])
    
        model_start = time.time()
        if not os.path.isfile(submodel_name):
        # level_0_model, level_1_model, indices = linear.train_tree_subsample(
        #     datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
        # print("training one model cost:", time.time()-model_start, flush=True)
        # #tmp, indices = linear.train_1vsrest_subsample(
        # #    datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate)
        # with open(submodel_name, "wb") as F:
        #     pickle.dump((level_0_model, level_1_model, indices), F, protocol=5)
            print(datasets["train"]["y"].shape, flush=True)
            models = []
            for idx in range(10):
            #model = linear.train_tree(
                model = linear.train_tree_partition(
                    datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q",K=ARGS.K)
                models += [model]
            with open(submodel_name, "wb") as F:
                pickle.dump(models, F, protocol=5)


print("training all models cost:", time.time()-start, flush=True)
