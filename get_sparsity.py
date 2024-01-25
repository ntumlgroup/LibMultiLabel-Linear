import warnings
import yaml

warnings.filterwarnings("ignore", message=".*")

from libmultilabel.linear.utils import load_pipeline
import matplotlib.pyplot as plt
from threshold_experiment import write_file
from collections import defaultdict 
from typing import Any
import os
import argparse
import numpy as np
import json
import pdb
import re

def _tree_sparsity():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--root_path")
    parser.add_argument("--l1_name", help="output models")
    parser.add_argument("--l2_name", help="model name")

    args = parser.parse_args()
    l1_name, l2_name, root_path = args.l1_name, args.l2_name, args.root_path
    
    preprocessor, l1_model = load_pipeline(os.path.join(root_path, l1_name, "linear_pipeline.pickle"))
    preprocessor, l2_model = load_pipeline(os.path.join(root_path, l2_name, "linear_pipeline.pickle"))

    l1 = len(l1_model.flat_model.weights.data)
    l2 = len(l2_model.flat_model.weights.data)

    print("l1_NNZ: ", l1)
    print("l2_NNZ: ", l2)


#loading L2
def _1vsrest_sparsity_l2():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--baseline_path", help="baseline model path")
    parser.add_argument("--threshold_path", help="threshold model")

    args = parser.parse_args()
    baseline, threshold = args.baseline_path, args.threshold_path

    preprocessor, base_model = load_pipeline(os.path.join(baseline, "linear_pipeline.pickle"))
    preprocessor, thresh_model = load_pipeline(threshold)

    wb = np.abs(base_model.weights)
    wt = np.abs(thresh_model.weights)

    base_nnz = np.sum(wb != 0)
    thresh_nnz = np.sum(wt != 0)

    threshold_path = threshold.rsplit("/", 2)
    log_path = os.path.join(threshold_path[0], "logs.txt")
    print("LOG_PATH", log_path)
    with open(log_path, "a") as f:
        f.write(threshold_path[1] + "\nbase_nnz: " + str(base_nnz) + " thresh_nnz: " + str(thresh_nnz) + "\n\n")

    # print("base_nnz: ", base_nnz, "thresh_nnz: ", thresh_nnz)

#loading l1
def _1vsrest_sparsity_l1():
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--baseline_path", help="baseline model path")

    args = parser.parse_args()
    baseline = args.baseline_path
    preprocessor, base_model = load_pipeline(os.path.join(baseline, "linear_pipeline.pickle"))
    data = np.abs(base_model.weights)

    baseline_path = baseline.rsplit("/", 2)
    print(baseline_path)
    log_path = os.path.join(baseline_path[0], "logs.txt")

    write_file(data, baseline_path[1])

def dualsprase(path):
    filename = os.listdir(path)
    ds_dict = defaultdict(dict)

    for f in filename:
        with open('logd.txt', 'a') as fp:
            fp.write(f + '\n')

        if 'log' in f: 
            continue
        preprocessor, model = load_pipeline(os.path.join(path, f, 'linear_pipeline.pickle'))

        data = np.abs(model.weights)
        nnz =  np.sum(data != 0)
        
        pattern = "[0-9]+e-\d"
        tolerance = re.search(pattern, f).group()

        if '_mi30_npf' in f:
            ds_dict['_mi30_npf'][tolerance] = nnz
        elif '_mi300_npf' in f:
            ds_dict['_mi300_npf'][tolerance] = nnz
        elif '_mi300' in f:
            ds_dict['_mi300'][tolerance] = nnz
        elif '_mi30' in f:
            ds_dict['_mi30'][tolerance] = nnz
    
    ds_dict = dict(ds_dict)
    with open(os.path.join(path, 'nnz-jlogs.json'), 'w') as f:
        json.dump(ds_dict, f, default=str)


def graph_dualsparse(path, data):
    
    with open(os.path.join(path, 'nnz-jlogs.json'), "r") as f:
        c = json.load(f)

    _mi30_npf = {float(k):int(n) for k,n, in c['_mi30_npf'].items()}
    _mi300_npf = {float(k):int(n) for k,n, in c['_mi300_npf'].items()}
    _mi30 = {float(k):int(n) for k,n, in c['_mi30'].items()}
    _mi300 = {float(k):int(n) for k,n, in c['_mi300'].items()}

    _mi30_npf = dict(sorted(_mi30_npf.items()))
    _mi300_npf = dict(sorted(_mi300_npf.items()))
    _mi30 = dict(sorted(_mi30.items()))
    _mi300 = dict(sorted(_mi300.items()))


    plt.plot(list(_mi30.keys()), list(_mi30.values()), label='mi30') 
    plt.plot(list(_mi300.keys()), list(_mi300.values()), label='mi300') 
    plt.plot(list(_mi30_npf.keys()), list(_mi30_npf.values()), label='mi30_npf') 
    plt.plot(list(_mi300_npf.keys()), list(_mi300_npf.values()), label='mi300_npf') 
    plt.legend()
    plt.title(data + " dualsparse")
    plt.xlabel("tolerance")
    plt.ylabel("nnz")
    plt.xscale("log")
    plt.savefig(path)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, description="multi-label learning for text classification")
    parser.add_argument("--baseline_path", help="baseline model path")
    parser.add_argument("--data", help="baseline model path")

    
    args = parser.parse_args()
    baseline, data = args.baseline_path, args.data

    graph_dualsparse(baseline, data)
    # dualsprase(baseline)
    # _tree_sparsity()
    # _1vsrest_sparsity_l1()
    # _1vsrest_sparsity_l2()
    # with open(os.path.join(baseline, 'nnz-jlogs.txt'), 'r') as f:
    #     c = json.load(f)
    #     breakpoint()
    #     print(c)





    

