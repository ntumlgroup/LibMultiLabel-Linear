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
total_preds = np.zeros([datasets["test"]["x"].shape[0], datasets["train"]["y"].shape[1]])
total_cnts = np.zeros(datasets["train"]["y"].shape[1])

model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

print(model_name)

for model_idx in range(num_models):
    submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    if os.path.isfile(submodel_name):
        with open(submodel_name, "rb") as F:
            tmp = pickle.load(F)
        tmp, indices = tmp
    else:
        tmp, indices = linear.train_tree_subsample(
                datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
        with open(submodel_name, "wb") as F:
            pickle.dump((tmp, indices), F)

    preds = tmp.predict_values(datasets["test"]["x"], beam_width=ARGS.beam_width)
    for idx in range(len(indices)):
        total_preds[:, indices[idx]] += preds[:, idx]
        total_cnts[indices[idx]] += 1


target = datasets["test"]["y"].toarray()

total_preds /= num_models

metrics = linear.compute_metrics(
            total_preds,
            target,
            monitor_metrics=["P@1", "P@3", "P@5"],
            )

print("mean in all labels:", metrics)

total_preds *= num_models
total_preds /= total_cnts+1e-16

metrics = linear.compute_metrics(
            total_preds,
            target,
            monitor_metrics=["P@1", "P@3", "P@5"],
            )

print("mean in subsampled labels:", metrics)

print("Total time:", time.time()-training_start)
