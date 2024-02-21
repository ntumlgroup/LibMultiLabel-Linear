import warnings
warnings.filterwarnings("ignore", message=".*")
from libmultilabel.linear.utils import load_pipeline, save_pipeline_threshold_experiment
from libmultilabel.linear.utils import threshold_smart_indexing
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

if __name__ == "__main__":
    args = load_config()
    (threshold_options, model_path, result_path, dataset_name, linear_technique) = (
        args.threshold_options,
        args.model_path,
        args.result_path,
        args.dataset_name,
        args.linear_technique,
    )
    

    preprocessor, model = load_pipeline(os.path.join(model_path, "linear_pipeline.pickle"))
    model.weights = scipy.sparse.csr_matrix(model.weights)

    threshold_type, param = load_options(threshold_options)
    if not threshold_type:
        print("single threshold: ", param[-1])
        data = np.abs(model.weights)
        nnz = np.sum(data != 0)
        z = np.sum(data == 0)
        percentage = [param[-1]]
        data = np.ravel(data)
    else:
        # model.weights = scipy.sparse.csr_matrix(model.weights)
        if threshold_type == "geo":
            print("geometric_threshold: ", param)
            a, r, k, percentage = param[0], param[1], param[2], []
            percentage = [(1 - float(a) * (float(r) ** int(k))) for k in range(int(k))]
        elif "lin":
            print("quantile_threshold: ", param)
            percentage = [round(i * 0.01, 2) for i in range(0, 100)]
            print(percentage)
        data = np.abs(model.weights.data)

    if not os.path.isdir(os.path.join(result_path)):
        os.makedirs(os.path.join(result_path))

    if not os.path.isdir(os.path.join(result_path, "models")):
        os.makedirs(os.path.join(result_path, "models"))
        
    if not os.path.isdir(os.path.join(result_path, "logs")):
        os.makedirs(os.path.join(result_path, "logs"))

    thresh, i = np.quantile(data, percentage), 0

    for t in tqdm(thresh):
        try:
            new_model = threshold_smart_indexing(model, float(t), linear_technique)
        except ValueError:
            pickle.dump(scipy.sparse._compressed.args_, open('arguments2.pickle', 'wb'))
            raise

        fname = dataset_name.split("/")[-1] + "--" + str(round(float(percentage[i]), 5)) + "--" + str(t) + ".pickle"
        save_pipeline_threshold_experiment(
            os.path.join(result_path, "models"),
            preprocessor,
            new_model,
            filename=fname,
        )
        i += 1
