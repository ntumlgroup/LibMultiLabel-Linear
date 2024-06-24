import warnings

warnings.filterwarnings("ignore", message=".*")
import libmultilabel.linear as linear
import pruning
import scipy.sparse
from typing import Any
import os
import numpy as np
import argparse
from tqdm import tqdm
import copy
import json
import pdb


def load_config():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--threshold_options", default="geo/.15/.95/100", help="type of thresholding")
    parser.add_argument("--model_path", default="runs/", help="path to baseline model")
    parser.add_argument("--result_path", default="runs/", help="path to where models should be saved")
    parser.add_argument("--dataset_name")
    parser.add_argument("--linear_technique", default="1vsrest", help="1vsrest or tree")
    parser.add_argument("--threshold_technique", default=False)
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


def create_dirs(result_path):
    if not os.path.isdir(os.path.join(result_path)):
        os.makedirs(os.path.join(result_path))
    if not os.path.isdir(os.path.join(result_path, "models")):
        os.makedirs(os.path.join(result_path, "models"))
    if not os.path.isdir(os.path.join(result_path, "logs")):
        os.makedirs(os.path.join(result_path, "logs"))


if __name__ == "__main__":
    args = load_config()
    (threshold_options, model_path, result_path, dataset_name, linear_technique, threshold_technique) = (
        args.threshold_options,
        args.model_path,
        args.result_path,
        args.dataset_name,
        args.linear_technique,
        args.threshold_technique,
    )
    create_dirs(result_path)
    threshold_type, param = load_options(threshold_options)

    if not threshold_type:
        print("single threshold: ", param[-1])
    else:
        if threshold_type == "geo":
            print("geometric_threshold: ", param)
            a, r, k, quantiles = param[0], param[1], param[2], []
            quantiles = [(1 - float(a) * (float(r) ** int(k))) for k in range(int(k))]
        elif "lin":
            print("linear_threshold: ", param)
            quantiles = [round(i * 0.01, 2) for i in range(int(param[0]), int(param[1]))]

    nnz_dict = {}
    if threshold_technique == "thresh_node":
        preprocessor, model = linear.utils.load_pipeline(os.path.join(model_path, "linear_pipeline.pickle"))

        num_quantiles = 100
        thresholds = pruning.concat_thresholds(model, num_quantiles)
        for i in range(thresholds.shape[0]):
            model.flat_model.weights = linear.utils.threshold_by_label(
                model.flat_model.weights.tocsc(), thresholds[i, :]
            )
            model.flat_model.weights = model.flat_model.weights.tocsr()

            linear.utils.save_pipeline_threshold_experiment(
                os.path.join(result_path, "models"),
                preprocessor,
                model,
                filename=f"{str(quantiles[i])}.pickle",
            )
            nnz_dict[str(quantiles[i])] = str(np.sum(model.flat_model.weights != 0))

    elif threshold_technique == "thresh_fixed":
        preprocessor, model = linear.utils.load_pipeline(os.path.join(model_path, "linear_pipeline.pickle"))
        model.flat_model.weights = scipy.sparse.csr_matrix(model.flat_model.weights)

        thresh_log = []
        thresh_percentile = {}

        thresh, i = np.quantile(np.abs(model.flat_model.weights.data), quantiles), 0
        for t in tqdm(thresh):
            model.flat_model.weights = linear.utils.threshold_fixed(model.flat_model.weights, float(t))

            linear.utils.save_pipeline_threshold_experiment(
                os.path.join(result_path, "models"),
                preprocessor,
                model,
                filename=f"{str(round(float(quantiles[i]), 5))}.pickle",
            )
            nnz_dict[str(quantiles[i])] = {"nnz": str(np.sum(model.flat_model.weights != 0)), "threshold": str(t)}
            i += 1

    with open(os.path.join(result_path, "nnz.json"), "w") as file:
        json.dump(nnz_dict, file)
