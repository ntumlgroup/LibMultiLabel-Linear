from collections import defaultdict 
import numpy as np
import pruning
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import collections
import json
import os


def showstats(name, ts):
    modelpath = f"runs/approx_pruning/{name}"
    _, model = linear.load_pipeline(f"{modelpath}/linear_pipeline.pickle")
    nnz = model.flat_model.weights.nnz
    print(f"{model.flat_model.weights.nnz=}")
    print(f"{model.flat_model.weights.shape=}")
    print()

    def evaluate(t, pruned_weights):
        print(f"{t=}")
        print(f"{pruned_weights.nnz=}")
        print(f"ratio = {pruned_weights.nnz / nnz:.4f}")
        print()

    ts = np.array(ts)
    pruning.approx(model, ts, evaluate, 0)

def showmetrics(name, metrics, ts):
    ts = np.array(ts, dtype="d")
    modelpath = f"approx_pruning/{name}"
    pp, model = linear.load_pipeline(f"{modelpath}/linear_pipeline.pickle")
    dataformat = pp.data_format
    datapath = f"data/{name}"
    dataset = linear.load_dataset(dataformat, test_path=f"{datapath}/test.{dataformat}")
    dataset = pp.transform(dataset)

    morestats = {}
    def statscallback(t, pruned_weights):
        morestats[t] = pruned_weights.nnz
    
    nnz = model.flat_model.weights.nnz
    results = pruning.approx_evaluate(model, dataset["test"]["y"], dataset["test"]["x"], metrics, ts, 1024, statscallback)

    metrics = []
    for k, v in results.items():
        tab = {}
        tab["k"] = k
        tab["metrics"] = v
        tab["nnz"] = morestats[k]
        tab["ratio"] = morestats[k] / nnz
        metrics.append(tab)

    mode = "a"
    if not os.path.exists(f"approx_pruning/{name}/logs/"):
        os.makedirs(f"approx_pruning/{name}/logs/")
        mode = "w"

    all_metrics = {}
    all_metrics["all_metrics"] = metrics
    with open(f'approx_pruning/{name}/logs/amazoncat-13k-logs2.json', mode) as file:
        json.dump(all_metrics, file)

def load_graph(name, metrics):
    with open(f"approx_pruning/{name}/logs/{name}-logs2.json", "r") as f:
        data = json.load(f)["all_metrics"]
        metric_list = defaultdict(list)

        for d in data:
            for m in metrics:
                metric_list[m].append(d["metrics"][m])
            metric_list["t"].append(d["k"])

        for m in metric_list:
            if m != "t": 
                plt.scatter(metric_list["t"], metric_list[m],label=m, marker=".")
        
        plt.legend()
        plt.title(f'{name}: t vs scores')
        plt.xlabel("t")
        plt.ylabel("scores")
        plt.savefig(f"approx_pruning/{name}/{name}-approxpruning.png")
        plt.close()
                
    



    


if __name__ == "__main__":
    
    # showmetrics("eur-lex", ["P@1", "P@5", "NDCG@3", "NDCG@5", "R@5"], 10 * 1.05**np.arange(100))
    # showmetrics("rcv1", ["P@1", "P@5", "NDCG@3", "NDCG@5", "R@5"], 10 * 1.05**np.arange(100))
    showmetrics("amazoncat-13k/ver1", ["P@1", "P@5", "NDCG@3", "NDCG@5", "R@5"], 10 * 1.05**np.arange(100))  

    # load_graph("eur-lex", ["P@1", "P@5", "NDCG@3", "NDCG@5", "R@5"])
    # load_graph("rcv1", ["P@1", "P@5", "NDCG@3", "NDCG@5", "R@5"])