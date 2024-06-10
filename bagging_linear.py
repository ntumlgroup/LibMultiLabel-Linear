import libmultilabel.linear as linear
import time
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os
import time
import math
from tqdm import tqdm

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

def predict_in_batches(model_name, batch_size, model_idx):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    submodel_name = "./models/" + model_name + "-{}".format(model_idx)
    sub_start = time.time()
    with open(submodel_name, "rb") as F:
        tmp = pickle.load(F)
    level_0_model, level_1_model, indices = tmp
    print("model loaded:", time.time()-sub_start, flush=True)

    for i in range(num_batches):
        print("process batches id:", i, flush=True)
        pred_name = "./preds/" + model_name + "-{}".format(model_idx) + "_batch-idx-{}".format(i)
        if os.path.isfile(pred_name):
            continue
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
        sub_start = time.time()
        preds = level_1_model.predict_values(tmp_data, level_0_model, beam_width=ARGS.beam_width)
        #preds = tmp.predict_values(tmp_data)
        print("preds cost:", time.time()-sub_start, flush=True)
        sub_start = time.time()
        with open(pred_name, "wb") as F:
            pickle.dump(preds, F, protocol = 5)
        print("dump cost:", time.time()-sub_start, flush=True)

def metrics_in_batches_without_pred(model_name, batch_size):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])
    for i in range(num_batches):
        print("process batches id:", i, flush=True)
        start = time.time()
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]

        pred_name = "./preds/" + model_name + "-{}".format(0) + "_batch-idx-{}".format(i)
        sub_start = time.time()
        with open(pred_name, "rb") as F:
            main_preds = pickle.load(F)
        print("preds cost:", time.time()-sub_start, flush=True)

        #for model_idx in range(ARGS.num_models):
        for model_idx in range(1, 3):
            pred_name = "./preds/" + model_name + "-{}".format(model_idx) + "_batch-idx-{}".format(i)
            sub_start = time.time()
            with open(pred_name, "rb") as F:
                preds = pickle.load(F)
            print("preds cost:", time.time()-sub_start, flush=True)
            main_preds = np.concatenate((main_preds, preds), axis=1)
        print("preds shape = ", main_preds.shape, flush=True)

        sub_start = time.time()
        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        metrics.update(main_preds, target)
        print("cal metrics cost:", time.time()-sub_start, flush=True)
        #with open(model_name + "-metrics-batch-idx-{}".format(i), 'wb') as F:
        #    pickle.dump(metrics, F)
        #print("cost:", time.time()-start, flush=True)

    return metrics.compute()


def metrics_in_batches(model_name, batch_size):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])
    for i in range(num_batches):
        print("process batches id:", i, flush=True)
        start = time.time()
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
        total_preds = np.zeros([tmp_data.shape[0], datasets["train"]["y"].shape[1]], order='F')
        total_cnts = np.zeros(datasets["train"]["y"].shape[1])

        submodel_name = "./models/" + model_name + "-{}".format(0)
        sub_start = time.time()
        with open(submodel_name, "rb") as F:
            total_models = pickle.load(F)
        for model_idx in range(10):
            models, metalabels = total_models[model_idx]
            #model = tmp
            print("model loaded:", time.time()-sub_start, flush=True)
        
            sub_start = time.time()
            # preds = level_1_model.predict_values(tmp_data, level_0_model, beam_width=ARGS.beam_width)
            for metalabel in tqdm(range(len(models))):
                model = models[metalabel]
                preds = model.predict_values(tmp_data, beam_width=ARGS.beam_width)
                preds = preds.toarray(order='F')
                indices = metalabels == metalabel
                total_preds[:, indices] += preds
                total_cnts[indices] += 1
            print("preds cost:", time.time()-sub_start, flush=True)
            #sub_start = time.time()

        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        print("min total_cnts = ", np.min(total_cnts), flush=True)
        total_preds /= total_cnts+1e-16
        metrics.update(total_preds, target)
        print("cost:", time.time()-start, flush=True)

    return metrics.compute()

    # # for amazon-670k full only
    # submodel_name = "./models/" + model_name + "-{}".format(0)
    # sub_start = time.time()
    # 
    # with open(submodel_name, "rb") as F:
    #     tmp = pickle.load(F)
    # tmp, indices = tmp
    # print("model loaded:", time.time()-sub_start, flush=True)
    # for i in range(num_batches):
    #     print("process batches id:", i, flush=True)
    #     start = time.time()
    #     tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
    #     total_preds = np.zeros([tmp_data.shape[0], datasets["train"]["y"].shape[1]], order='F')
    #     total_cnts = np.zeros(datasets["train"]["y"].shape[1])
    #         
    #     sub_start = time.time()
    #     preds = tmp.predict_values(tmp_data, beam_width=ARGS.beam_width)
    #     print("preds cost:", time.time()-sub_start, flush=True)
    #     sub_start = time.time()
    #     preds = preds.toarray(order='F')
    #     total_preds[:, indices] += preds
    #     total_cnts[indices] += 1
    #     print("add preds cost:", time.time()-sub_start, flush=True)

    #     target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
    #     total_preds /= total_cnts+1e-16
    #     metrics.update(total_preds, target)
    #     print("cost:", time.time()-start, flush=True)

    # return metrics.compute()


#model_name = "Rand-label-Forest-LD_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
#model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
model_name = "Rand-label-Forest-No-replacement_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )
# model_name = "OVR_{data}_seed={seed}.model".format(
#         seed = ARGS.seed,
#         data = os.path.basename(ARGS.datapath)
#         )

metrics = metrics_in_batches(model_name, 10000)
# for amazon-3m
#metrics = metrics_in_batches_without_pred(model_name, 10000)
#metrics = metrics_in_batches(model_name, 1000)
#predict_in_batches(model_name, 10000, ARGS.idx)


print("mean in subsampled labels:", metrics)

print("Total time:", time.time()-training_start)
