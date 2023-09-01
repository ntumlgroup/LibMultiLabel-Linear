import libmultilabel.linear as linear
import time
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os
import time
import math

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
def predict_in_batches(model_name, batch_size, model_idx):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    if model_idx >= 0:
        submodel_name = "./models/" + model_name + "-{}".format(model_idx)
        sub_start = time.time()
        with open(submodel_name, "rb") as F:
            tmp = pickle.load(F)
        tmp, indices = tmp
        print("{}th model loaded:".format(model_idx), time.time()-sub_start, flush=True)
    
        for i in range(num_batches):
            pred_name = "./preds/" + model_name.split(".model")[0] + "-{}".format(model_idx) + "_batch-idx-{}".format(i) 
            if os.path.isfile(pred_name):
                continue

            sub_start = time.time()
            print("process batches id:", i, flush=True)
            start = time.time()
            tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
            preds = tmp.predict_values(tmp_data, beam_width=ARGS.beam_width)
            print("sub cost:", time.time()-sub_start, flush=True)

            print("dump {}th preds".format(model_idx), flush=True)
            sub_start = time.time()
            with open(pred_name, "wb") as F:
                pickle.dump((preds, indices), F)
            print("dump cost:", time.time()-sub_start, flush=True)

    else:
        for model_idx in range(ARGS.num_models):
            submodel_name = "./models/" + model_name + "-{}".format(model_idx)
            sub_start = time.time()
            with open(submodel_name, "rb") as F:
                tmp = pickle.load(F)
            tmp, indices = tmp
            print("{}th model loaded:".format(model_idx), time.time()-sub_start, flush=True)
        
            for i in range(num_batches):
                pred_name = "./preds/" + model_name.split(".model")[0] + "-{}".format(model_idx) + "_batch-idx-{}".format(i) 
                if os.path.isfile(pred_name):
                    continue

                sub_start = time.time()
                print("process batches id:", i, flush=True)
                start = time.time()
                tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
                preds = tmp.predict_values(tmp_data, beam_width=ARGS.beam_width)
                print("sub cost:", time.time()-sub_start, flush=True)

                print("dump {}th preds".format(model_idx), flush=True)
                sub_start = time.time()
                with open(pred_name, "wb") as F:
                    pickle.dump((preds, indices), F)
                print("dump cost:", time.time()-sub_start, flush=True)


def metrics_in_batches(model_name, batch_size):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])
    for i in range(num_batches):
        print("process batches id:", i, flush=True)
        start = time.time()
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
        total_preds = np.zeros([tmp_data.shape[0], datasets["train"]["y"].shape[1]])
        total_cnts = np.zeros(datasets["train"]["y"].shape[1])
        for model_idx in range(ARGS.num_models):
            submodel_name = "./models/" + model_name + "-{}".format(model_idx)
            sub_start = time.time()
            
            with open(submodel_name, "rb") as F:
                tmp = pickle.load(F)
            tmp, indices = tmp
            print("model loaded:", time.time()-sub_start, flush=True)
        
            sub_start = time.time()
            preds = tmp.predict_values(tmp_data, beam_width=ARGS.beam_width)
            for idx in range(len(indices)):
                total_preds[:, indices[idx]] += preds[:, idx]
                total_cnts[indices[idx]] += 1
            print("sub cost:", time.time()-sub_start, flush=True)

        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        total_preds /= total_cnts+1e-16
        metrics.update(total_preds, target)
        print("cost:", time.time()-start, flush=True)

    return metrics.compute()

model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

#metrics = metrics_in_batches(model_name, 10000)
#metrics = metrics_in_batches(model_name, 1000)
predict_in_batches(model_name, 10000, ARGS.idx)

#print("mean in subsampled labels:", metrics)

print("Total time:", time.time()-training_start)
