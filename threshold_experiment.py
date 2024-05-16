import warnings
warnings.filterwarnings("ignore", message=".*")

import scipy.sparse
from typing import Any
import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pdb
import pickle



def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def load_config():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--threshold_options", default="geo/.15/.95/100", help="type of thresholding")
    parser.add_argument("--model_path", default="runs/", help="path to baseline model")
    parser.add_argument("--result_path", default="runs/", help="path to where models should be saved")
    parser.add_argument("--dataset_name")
    parser.add_argument("--linear_technique", default="1vsrest", help="1vsrest or tree")
    parser.add_argument("--thresh_node", default=False)
    args = parser.parse_args()
    return args


def load_options(threshold_options):
    threshold_options = threshold_options.split("/")
    print(threshold_options)
    if len(threshold_options) == 1:
        return None, np.float_(threshold_options)
    elif threshold_options[0] == "geo":
        return "geo", threshold_options[1:]
    elif threshold_options[0] == "lin":
        return "lin", threshold_options[1:]

def write_file(data, dataset_name):
    nnz = np.sum(data != 0)
    z = np.sum(data == 0)
    with open("threshold_logs.txt", "a") as f:
        f.write(
            dataset_name
            + "\n"
            + str(data.shape)
            + " ("
            + str(int(data.shape[0]) * int(data.shape[1]))
            + ")"
            + "\n"
            + "NNZ: "
            + str(nnz)
            + " Z: "
            + str(z)
            + "\n"
        )
def create_dirs(result_path):
    if not os.path.isdir(os.path.join(result_path)):
        os.makedirs(os.path.join(result_path))

    if not os.path.isdir(os.path.join(result_path, "models")):
        os.makedirs(os.path.join(result_path, "models"))
        
    if not os.path.isdir(os.path.join(result_path, "logs")):
        os.makedirs(os.path.join(result_path, "logs"))

if __name__ == "__main__":
    args = load_config()
    
    (threshold_options, model_path, result_path, dataset_name, linear_technique, thresh_node) = (
        args.threshold_options,
        args.model_path,
        args.result_path,
        args.dataset_name,
        args.linear_technique,
        args.thresh_node,
    )
    create_dirs(result_path)
    threshold_type, param = load_options(threshold_options)

    if not threshold_type:
        print("single threshold: ", param[-1])
    else:
        if threshold_type == "geo":
            print("geometric_threshold: ", param)
            a, r, k, percentage = param[0], param[1], param[2], []
            percentage = [(1 - float(a) * (float(r) ** int(k))) for k in range(int(k))]
        elif "lin":
            print("quantile_threshold: ", param)
            percentage = [round(i * 0.01, 2) for i in range(0, 100)]
            print(percentage)
        
    
    if thresh_node:
        import libmultilabel.linear as linear

        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("txt", f"data/{dataset_name}/train.txt", f"data/{dataset_name}/test.txt")
        dataset = preprocessor.fit_transform(dataset)

        root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"])
        
        model = linear.tree.train_tree_thresh(
            dataset["train"]["y"],
            dataset["train"]["x"],
            root,
            quantile=percentage) # by node train + thresh
        
        i = 0
        for t in tqdm(percentage):
            def thresh_node(node):
                for i in tqdm(range(node.model.weights.shape[1])):
                    col = node.model.weights[:,i]
                    abs_weights = np.abs(col.data)

                    threshold = np.quantile(abs_weights, quantile=percentage)
                    idx = abs_weights < threshold

                    node.model.weights[idx, i] = 0
                    node.model.weights.eliminate_zeros()

            root.dfs(thresh_node)
            flat_model, weight_map = linear.tree._flatten_model(root)
            thresh_model = linear.tree.TreeModel(root, flat_model, weight_map)
        
            fname = dataset_name.split("/")[-1] + "--" + str(round(float(percentage[i]), 5)) + "--" + str(t) + ".pickle"

            linear.utils.save_pipeline_threshold_experiment(
                os.path.join(result_path, "models"),
                preprocessor,
                percentage,
                filename=fname,
            )
            i += 1

    else: 
        
        from libmultilabel.linear.utils import load_pipeline, save_pipeline_threshold_experiment
        from libmultilabel.linear.utils import threshold_smart_indexing   

        preprocessor, model = linear.utils.load_pipeline(os.path.join(model_path, "linear_pipeline.pickle"))
        model.weights = scipy.sparse.csr_matrix(model.weights)
        data = np.abs(model.weights.data)
        thresh, i = np.quantile(data, percentage), 0
        for t in tqdm(thresh):
            new_model = threshold_smart_indexing(model, float(t), linear_technique)
            fname = dataset_name.split("/")[-1] + "--" + str(round(float(percentage[i]), 5)) + "--" + str(t) + ".pickle"

            save_pipeline_threshold_experiment(
                os.path.join(result_path, "models"),
                preprocessor,
                new_model,
                filename=fname,
            )
            i += 1
