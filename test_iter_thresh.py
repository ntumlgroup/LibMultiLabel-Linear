import numpy as np
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import pruning
import datetime
import time
import json


if __name__ == "__main__":    
    datasets = [
        "rcv1",
        "eur-lex",
        "wiki10-31k",
        "amazoncat-13k/ver1"
    ]
    for d in datasets:
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
        dataset = preprocessor.fit_transform(dataset)

        start = time.perf_counter()
        model, q = pruning.iter_thresh(d, dataset, initial_quantile=0.80, perf_drop_tolerance=.007, K=100, dmax=10, option="-s 1 -e 0.0001 -c 1")        
        end = time.perf_counter()
        
        with open("runs/logs/iter_thresh_train_times.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} {d}: {str(end-start)}")

        linear.utils.save_pipeline_threshold_experiment(
            f'runs/iter_thresh/{d}/models',
            preprocessor,
            model,
            filename=f'{q}.pickle',
        )