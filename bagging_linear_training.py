import libmultilabel.linear as linear
import time
import numpy as np
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

datasets = linear.load_dataset(
               "svm",
               os.path.join(ARGS.datapath, "train.svm"),
               os.path.join(ARGS.datapath, "test.svm"),
           )

preprocessor = linear.Preprocessor()
preprocessor.fit(datasets)
datasets = preprocessor.transform(datasets)

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

total_preds = np.zeros([datasets["test"]["x"].shape[0], datasets["train"]["y"].shape[1]])
total_cnts = np.zeros(datasets["train"]["y"].shape[1])

model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

if ARGS.idx >= 0:
    model_idx = ARGS.idx
    submodel_name = "./models/" + model_name + "-{}".format(model_idx)

    np.random.seed(seed_pool[model_idx])

    if not os.path.isfile(submodel_name):
        tmp, indices = linear.train_tree_subsample(
                datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
        with open(submodel_name, "wb") as F:
            pickle.dump((tmp, indices), F)

else:
    for model_idx in range(num_models):
        submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    
        np.random.seed(seed_pool[model_idx])
    
        if not os.path.isfile(submodel_name):
            tmp, indices = linear.train_tree_subsample(
                    datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
            with open(submodel_name, "wb") as F:
                pickle.dump((tmp, indices), F)
