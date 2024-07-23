import numpy as np
import libmultilabel.linear as linear
import libmultilabel.common_utils
import threshold_experiment
import importlib as fp
import matplotlib.pyplot as plt
import experiment_utils
import pruning
import datetime
import time
import sys
import ast

if __name__ == "__main__":
    # run_time = sys.argv[1]
    # datasets = ["rcv1", "eur-lex", "wiki10-31k", "amazoncat-13k/ver1"]
    datasets = ["amazoncat-670k"]
    threshold_methods = ["all_label", "per_label"]
    tolerance = 0.001
    quantiles = [(1 - (1 - 0) * (0.8**k)) for k in range(100)]  # adjust when needed or make input
    for threshold_method in threshold_methods:
        print(f"iter-thresh-global\nthreshold method: {threshold_method}\ntolerance: {tolerance}\n{datasets}\n")
        root_path = f"runs/iter-thresh-global/{threshold_method}/{str(tolerance)}-tolerance"
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        for d in datasets:
            path = f"{root_path}/{d}/{run_time}"
            datapath = f"data/{d}"
            preprocessor = linear.Preprocessor()
            if d == "mimic" or d == "amazoncat-670k":
                dataset = linear.load_dataset("txt", f"{datapath}/train.txt", f"{datapath}/test.txt")
            else:
                print(d)
                dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
            dataset = preprocessor.fit_transform(dataset)

            threshold_experiment.create_dirs(path)

            start = time.perf_counter()
            model, quant_selected, root = pruning.iter_thresh_global(
                dataset,
                initial_quantile=0,
                perf_drop_tolerance=tolerance,
                log_path=f"{path}/logs/run-logs.txt",
                threshold_method=threshold_method,
                K=100,
                dmax=10,
                option="-s 1 -e 0.0001 -c 1",
            )
            end = time.perf_counter()

            with open(f"{path}/train_times.txt", "a") as file:
                file.write(f"{datetime.datetime.now()} {d}: {str(end-start)} \n")

            with open("eval_metrics.txt", "r") as file:
                eval_metrics = file.read()
            eval_metrics = eval_metrics.split("/")

            start = time.perf_counter()
            metrics_collections = pruning.iter_thresh_evaluate(
                model, dataset["test"]["y"], dataset["test"]["x"], eval_metrics
            )
            end = time.perf_counter()

            with open(f"{path}/eval_times.txt", "a") as file:
                file.write(f"{datetime.datetime.now()} {d}: {str(end-start)} \n")

            linear.utils.save_pipeline_threshold_experiment(
                f"{path}/models",
                preprocessor,
                model,
                filename=f"{quant_selected}.pickle",
            )

            libmultilabel.common_utils.dump_log(
                metrics=metrics_collections,
                split="test",
                log_path=f"{path}/logs/iter-thresh.json",
            )
