import libmultilabel.linear as linear
import libmultilabel.common_utils
import threshold_experiment
import importlib as fp
import pruning
import datetime
import time
import sys
import ast

if __name__ == "__main__":
    run_time = sys.argv[1]
    # datasets = ["rcv1", "eur-lex", "wiki10-31k", "amazoncat-13k/ver1"]
    datasets = ["amazoncat-670k"]
    threshold_methods = ["all_label", "per_label"]
    tolerance = 0.001
    quantiles = [(1 - (1 - 0) * (0.8**k)) for k in range(100)]  # adjust when needed or make input
    with open("eval_metrics.txt", "r") as file:
        metrics = file.read()
    metrics = metrics.split("/")
    for threshold_method in threshold_methods:
        print(f"iter-thresh-retrain\nthreshold method: {threshold_method}\ntolerance: {tolerance}\n{datasets}\n")
        root_path = f"runs/iter-thresh-retrain/{threshold_method}/{str(tolerance)}-tolerance"
        # run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        for d in datasets:
            path = f"{root_path}/{d}/{run_time}"
            datapath = f"data/{d}"
            preprocessor = linear.Preprocessor()
            if d == "mimic":
                dataset = linear.load_dataset("txt", f"{datapath}/train_valid.txt", f"{datapath}/test.txt")
            else:
                dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
            dataset = preprocessor.fit_transform(dataset)

            if d == "amazoncat-13k/ver1":
                d = "amazoncat-13k"
            threshold_experiment.create_dirs(path)

            start = time.perf_counter()
            model, metrics_collection, eval_time = pruning.iter_thresh_retrain(
                dataset,
                initial_quantile=0.0,
                perf_drop_tolerance=tolerance,
                log_path=f"{path}/logs/run-logs.txt",
                threshold_method=threshold_method,
                K=100,
                dmax=10,
                option="-s 1 -e 0.0001 -c 1",
                metrics=metrics,
            )
            end = time.perf_counter()

            with open(f"{path}/train_times.txt", "a") as file:
                file.write(f"{datetime.datetime.now()} {d}: {str(end-start)} \n")
            with open(f"{path}/eval_times.txt", "a") as file:
                file.write(f"{datetime.datetime.now()} {d}: {str(eval_time)} \n")

            linear.utils.save_pipeline_threshold_experiment(
                f"{path}/models",
                preprocessor,
                model,
                filename=f"iter-thresh-retrain.pickle",
            )

            libmultilabel.common_utils.dump_log(
                metrics=metrics_collection,
                split="test",
                log_path=f"{path}/logs/iter-thresh-retrain.json",
            )
