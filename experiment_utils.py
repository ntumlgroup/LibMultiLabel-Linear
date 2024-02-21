import warnings
warnings.filterwarnings("ignore", message=".*")
import os
import json
from libmultilabel.linear.utils import load_pipeline, save_pipeline_threshold_experiment
# from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict 
import matplotlib.pyplot as plt
import argparse
import pdb
import re
import numpy as np


def create_graph(log_dir, metrics):
    thresh_list = [] # x
    metrics_score = defaultdict(list) # y
    metrics = metrics.split("/")
    for filename in os.listdir(log_dir):
        # if ".json" not in filename or "linear" in filename:
        #     continue

        thresh = filename.rsplit("--",1)[-1]
        if '.pickle.json' not in thresh or 'linear' in thresh:
            continue
        thresh = float(thresh.replace(".pickle.json", ""))
        pattern = "\d\.\d{6,}"

        # thresh = re.search(pattern, filename)
        # if not thresh:
        #     continue
        # thresh = thresh.group()
        thresh_list.append(float(thresh))
        with open(os.path.join(log_dir, filename), "r") as f:
            c = json.load(f)

            for m in metrics:
                metrics_score[m].append((thresh, float(c["test"][0][m])))

    for m in metrics_score: 
        print(m)
        x,y = zip(*metrics_score[m])
        plt.scatter(x, y,label=m, marker=".")

    model_dir = log_dir.rsplit('/', 1)[0]
    data_name  = model_dir.split('/')
    if not os.path.isdir(os.path.join(model_dir, "graphs")):
        os.makedirs(os.path.join(model_dir, "graphs"))
    plt.legend()
    plt.title(f'{data_name[-2]} {data_name[-1]} threshold vs scores')
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xlabel("threshold")
    plt.ylabel("scores")
    plt.savefig(os.path.join(model_dir, "graphs", "thresh_score.png"))
    plt.close()
            
    

        



def parse_folder(input_path, size_dict):
    """Function to parse log files from threshold experiment results.

    Args:
        input_path (str): path to the runs folder containing logs of prediction results for each threshold experiment
        size_dict (dict): dictionary that stores the size of each threshold model (KEY: threshold value, VALUE: model size)
        
    """
    s,i = [],0
    for filename in os.listdir(input_path):
        if "baseline" in filename:
            continue
        with open(os.path.join(input_path, filename, "logs.json"), "r") as j:
            # print(input_path, filename, 'logs.json')
            c = json.load(j)

            if c["config"]["checkpoint_path"] is not None:
                name = c["config"]["checkpoint_path"].split("/")
                name = name[-1].split(".pickle")
                # print(name)
            else:
                name = [0.0]
            if float(name[0]) in size_dict:
                if float(name[0]) > 7:
                    continue
                i += 1
                p1, p2, p3 = c["test"][0]["P@1"], c["test"][0]["P@3"], c["test"][0]["P@5"]
                s.append([float(name[0]), float(p1), float(p2), float(p3), float(size_dict[float(name[0])])])
    return s


def get_size(input_path):
    """Get the size of each experiment model

    Args:
        input_path (str): Path to folder containing all experimental threshold models
       
    """
    size_lookup = {}
    for file in os.listdir(input_path):
        s = file.split(".pickle")
        size_lookup[float(s[0])] = os.path.getsize(os.path.join(input_path, file))
    return size_lookup


def write_test(log_path):

    with open(os.path.join(log_path), "r") as j:
        logs = json.load(j)

    eval_log_path = log_path.rsplit('/', 1)

    with open(os.path.join(eval_log_path[0], 'eval_logs.txt'), 'a') as f:
        f.write(eval_log_path[1] + "\n")
        json.dump(logs["test"][0], f, indent=2)
        f.write("\n")
        # f.write(log_path + "\n" + str(logs["test"]) + "\n\n")

def metric_loader(log_path, metrics):
    if '.json' not in log_path:
        return

    metrics = metrics.split("/")
    with open(log_path, "r") as j:
            c = json.load(j)

    dic  = c["test"][0]

    for key in dic:
        dic[key] = round(dic[key], 5)


    vals = ""

    for n in metrics:
        if n not in dic:
            vals += "--" + " & "
        else:
            vals += (str(dic[n]) + " & ")
    
    data_name = log_path.split("/")
    with open(os.path.join("latex-logs.txt"), 'a') as f:
        f.write(str(metrics)+ "\n" + data_name[-1]+"\n"+vals+"\n")
    print(metrics)
    print(vals)

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
    (threshold_options, model_path, result_path, dataset_name, linear_technique, log_path, log_name, function_name, metrics) = (
        args.threshold_options,
        args.model_path,
        args.result_path,
        args.dataset_name,
        args.linear_technique,
        args.log_path,
        args.log_name,
        args.function_name,
        args.metrics
    )

    if function_name == 'metric_loader':
        metric_loader(log_path, metrics)
    elif function_name == 'create_graph':
        create_graph(log_path, metrics)
    elif log_name is not None and log_path is not None:
        write_test(log_path)
    
    