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

# OVR with bagging in instances
#
# model = linear.train_1vsrest_negative_sampling(
#             datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=0.1)
# for _ in range(num_models-1):
#     tmp = linear.train_1vsrest_negative_sampling(
#             datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=0.1)
#     model.weights += tmp.weights
#     model.bias += tmp.bias
# 
# model.weights /= num_models
# model.bias /= num_models
# 
# preds = linear.predict_values(model, datasets["test"]["x"])

num_models = ARGS.num_models

seed_pool = []
while len(seed_pool) != num_models:
    seed = np.random.randint(2**31 - 1)
    if seed not in seed_pool:
        seed_pool += [seed]

model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

if ARGS.idx >= 0:
    model_idx = ARGS.idx
    np.random.seed(seed_pool[model_idx])

    submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    if not os.path.isfile(submodel_name):
        tmp, indices = linear.train_tree_subsample(
                datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
        #tmp, indices = linear.train_1vsrest_subsample(
        #        datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate)
        with open(submodel_name, "wb") as F:
            pickle.dump((tmp, indices), F)

    # predict_name = "./preds/" + model_name.split(".model")[0] + "-{}".format(model_idx)
    # if not os.path.isfile(predict_name):
    #     num_instances = datasets["test"]["x"].shape[0]
    #     num_batches = math.ceil(num_instances / batch_size)
    #     preds = []
    #     for i in range(num_batches):
    #         tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
    #         preds += [ tmp.predict_values(tmp_data, beam_width=ARGS.beam_width) ]

    #     preds = sparse.vstack(preds)
    #     with open(predict_name, "wb") as F:
    #         pickle.dump((preds, indices), F)

else:
    for model_idx in range(num_models):
        model_start = time.time()
        submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    
        np.random.seed(seed_pool[model_idx])
    
        if not os.path.isfile(submodel_name):
            tmp, indices = linear.train_tree_subsample(
                    datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
            # tmp, indices = linear.train_1vsrest_subsample(
            #         datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate)
            with open(submodel_name, "wb") as F:
                pickle.dump((tmp, indices), F)
        print("training one model cost:", time.time()-model_start, flush=True)

print("training all models cost:", time.time()-start, flush=True)
