import numpy as np
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import experiment_utils
import pruning
import datetime
import time
import json


if __name__ == "__main__":
    datasets = ["rcv1", "eur-lex", "wiki10-31k", "amazoncat-13k/ver1"]
    quantiles = [(1 - (1 - 0) * (0.8**k)) for k in range(100)]  # adjust when needed or make input

    for d in datasets:
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
        dataset = preprocessor.fit_transform(dataset)

        start = time.perf_counter()
        model, quantile_idx = pruning.iter_thresh_global(
            d,
            dataset,
            initial_quantile=0,
            perf_drop_tolerance=0.01,
            K=100,
            dmax=10,
            option="-s 1 -e 0.0001 -c 1",
        )
        end = time.perf_counter()

        with open("runs/logs/iter_thresh_global_train_times.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} {d}: {str(end-start)} \n")

        linear.utils.save_pipeline_threshold_experiment(
            f"runs/iter_thresh_global/{d}/models",
            preprocessor,
            model,
            filename=f"{quantiles[quantile_idx]}.pickle",
        )
