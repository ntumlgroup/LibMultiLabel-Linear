import warnings

warnings.filterwarnings("ignore", message=".*")
import os
import json
from libmultilabel.linear.utils import load_pipeline, save_pipeline_threshold_experiment

# from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict
import matplotlib.pyplot as plt
import libmultilabel.linear as linear
import argparse
import pdb
import re
import numpy as np


def metric_loader(log_path, metrics):
    if ".json" not in log_path:
        return

    metrics = metrics.split("/")
    with open(log_path, "r") as j:
        c = json.load(j)
        dic = c["test"][0]

    for key in dic:
        dic[key] = round(dic[key], 5)

    vals = ""
    for n in metrics:
        if n not in dic:
            vals += "--" + " & "
        else:
            vals += str(dic[n]) + " & "

    data_name = log_path.split("/")
    with open(os.path.join("latex-logs.txt"), "a") as f:
        f.write(str(metrics) + "\n" + data_name[-1] + "\n" + vals + "\n")
    print(metrics)
    print(vals)


def create_graph(name, result_dir, metrics):
    if not os.path.isdir(os.path.join(result_dir, "graphs")):
        os.makedirs(os.path.join(result_dir, "graphs"))

    metrics_score = defaultdict(list)  # y
    metrics = metrics.split("/")

    # quantile vs score
    for filename in os.listdir(os.path.join(result_dir, "logs")):

        thresh = float(filename.replace(".json", ""))

        with open(os.path.join(result_dir, "logs", filename), "r") as f:
            c = json.load(f)

            for m in metrics:
                if "validation" in c:
                    metrics_score[m].append((thresh, float(c["validation"][0][m])))
                else:
                    metrics_score[m].append((thresh, float(c["test"][0][m])))

    for m in metrics_score:
        print(m)
        x, y = zip(*metrics_score[m])
        plt.scatter(x, y, label=m, marker=".")

    plt.legend()
    plt.title(f"{name} quantile vs scores")
    plt.xlabel("quantile (%)")
    plt.grid()
    plt.ylabel("scores")
    plt.savefig(os.path.join(result_dir, "graphs", f"quantile_score.png"))
    plt.close()

    # nnz vs score
    for filename in os.listdir(os.path.join(result_dir, "logs")):
        thresh = filename.replace(".json", "")
        with open(os.path.join(result_dir, "nnz.json")) as file:
            nnz = json.load(file)

        with open(os.path.join(result_dir, "logs", filename), "r") as f:
            c = json.load(f)
            for m in metrics:
                metrics_score[m].append((nnz[thresh], float(c["validation"][0][m])))

    for m in metrics_score:
        print(m)
        x, y = zip(*metrics_score[m])
        plt.scatter(x, y, label=m, marker=".")

    plt.legend()
    plt.title(f"{name} nnz vs scores")
    plt.xlabel("nnz")
    plt.ylabel("scores")
    plt.savefig(os.path.join(result_dir, "graphs", "nnz_score.png"))
    plt.close()


def node_predict(data, model_path, log_path, metrics):
    from sklearn.model_selection import train_test_split
    import libmultilabel.common_utils as common_utils
    import pruning

    datapath = f"data/{data}"
    preprocessor = linear.Preprocessor()
    dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
    dataset = preprocessor.fit_transform(dataset)
    np.random.seed(1337)
    x_tr, x_val, y_tr, y_val = train_test_split(
        dataset["train"]["x"], dataset["train"]["y"], test_size=0.2, train_size=0.8, random_state=2
    )

    preprocessor, model = load_pipeline(f"{model_path}")
    filename = model_path.split("/")[-1][:-7]
    metrics = metrics.split("/")
    metric_collection = pruning.iter_thresh_evaluate(model, y_val, x_val, metrics)
    common_utils.dump_log(metrics=metric_collection, split="test", log_path=f"{log_path}/{filename}.json")


def train_tree_compute_thresholds():
    import libmultilabel.linear as linear
    import libmultilabel.common_utils as common_utils
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pruning

    result_path = "/mnt/HDD-Seagate-16TB-2/justinchanchan8/tree-prunning-results/runs/tree"
    metrics = [
        "P@1",
        "P@2",
        "P@3",
        "P@4",
        "P@5",
        "P@6",
        "P@7",
        "P@8",
        "P@9",
        "P@10",
        "P@11",
        "P@12",
        "P@13",
        "P@14",
        "P@15",
        "P@16",
        "P@17",
        "P@18",
        "P@19",
        "P@20",
        "NDCG@1",
        "NDCG@2",
        "NDCG@3",
        "NDCG@4",
        "NDCG@5",
        "NDCG@6",
        "NDCG@7",
        "NDCG@8",
        "NDCG@9",
        "NDCG@10",
        "NDCG@11",
        "NDCG@12",
        "NDCG@13",
        "NDCG@14",
        "NDCG@15",
        "NDCG@16",
        "NDCG@17",
        "NDCG@18",
        "NDCG@19",
        "NDCG@20",
        "R@5",
        "R@10",
        "R@20",
        "R@50",
        "R@75",
        "R@100",
    ]
    datasets = ["rcv1", "eur-lex", "wiki10-31k", "amazoncat-13k/ver1"]
    datasets = ["amazoncat-13k/ver1"]
    for d in datasets:
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
        dataset = preprocessor.fit_transform(dataset)

        np.random.seed(1337)
        root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"], K=100, dmax=10)
        x_tr, x_val, y_tr, y_val = train_test_split(
            dataset["train"]["x"], dataset["train"]["y"], test_size=0.2, train_size=0.8, random_state=2
        )
        quantiles = [(1 - float(0.15) * (float(0.8) ** int(k))) for k in range(100)]
        model = linear.tree.train_tree_compute_threshold(y_tr, x_tr, root, quantiles, options="-s 1 -e 0.0001 -c 1")

        linear.utils.save_pipeline_threshold_experiment(
            f"{result_path}/{d}/L2/85-100-geometric-node-tr-val-split",
            preprocessor,
            model,
            filename="linear_pipeline.pickle",
        )

        metric_collection = pruning.iter_thresh_evaluate(model, y_val, x_val, metrics)
        common_utils.dump_log(
            metrics=metric_collection,
            split="test",
            log_path=f"{result_path}/{d}/L2/85-100-geometric-node-tr-val-split/logs.json",
        )


def load_config():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--threshold_options", default="geo/.15/.95/100", help="type of thresholding")
    parser.add_argument("--model_path", default="runs/", help="path to baseline model")
    parser.add_argument("--result_path", default="runs/", help="path to where models should be saved")
    parser.add_argument("--dataset_name")
    parser.add_argument("--linear_technique", default="1vsrest", help="1vsrest or tree")
    parser.add_argument("--log_path", default="log_path", help="log_path")
    parser.add_argument("--log_name", default="log_path", help="log_path")
    parser.add_argument("--function_name", default="log_path", help="log_path")
    parser.add_argument("--metrics", default="metrics", help="metrics")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_config()
    (
        threshold_options,
        model_path,
        result_path,
        dataset_name,
        linear_technique,
        log_path,
        log_name,
        function_name,
        metrics,
    ) = (
        args.threshold_options,
        args.model_path,
        args.result_path,
        args.dataset_name,
        args.linear_technique,
        args.log_path,
        args.log_name,
        args.function_name,
        args.metrics,
    )

    if function_name == "metric_loader":
        metric_loader(log_path, metrics)
    elif function_name == "create_graph":
        create_graph(dataset_name, log_path, metrics)
    elif function_name == "thresh_node_predict":
        node_predict(dataset_name, model_path, log_path, metrics)
    elif function_name == "train_tree_compute_thresholds":
        train_tree_compute_thresholds()
