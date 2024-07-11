import libmultilabel.linear as linear
import libmultilabel.common_utils
import threshold_experiment
import experiment_utils
import importlib as fp
import experiment_utils
import pruning
import datetime
import time


if __name__ == "__main__":
    with open("eval_metrics.txt", "r") as file:
        metrics = file.read()

    metrics = metrics.split("/")
    threshold_method = "fixed-threshold"
    datasets = ["rcv1", "eur-lex", "wiki10-31k", "amazoncat-13k/ver1"]
    quantiles = [(1 - (1 - 0) * (0.8**k)) for k in range(100)]  # adjust when needed or make input
    for d in datasets:
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("svm", f"{datapath}/train.svm", f"{datapath}/test.svm")
        dataset = preprocessor.fit_transform(dataset)

        if d == "amazoncat-13k/ver1":
            d = "amazoncat-13k"
        threshold_experiment.create_dirs(f"runs/iter-thresh-retrain/{threshold_method}/{d}")

        start = time.perf_counter()
        model, metrics_collection, eval_time = pruning.iter_thresh_retrain(
            dataset,
            initial_quantile=0.0,
            perf_drop_tolerance=0.001,
            log_path=f"runs/iter-thresh-retrain/{threshold_method}/{d}/logs/{d}-run-logs.txt",
            threshold_method="fixed",
            K=100,
            dmax=10,
            option="-s 1 -e 0.0001 -c 1",
            metrics=metrics,
        )
        end = time.perf_counter()

        with open(f"runs/iter-thresh-retrain/{threshold_method}/{d}/train_times.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} {d}: {str(end-start)} \n")
        with open(f"runs/iter-thresh-retrain/{threshold_method}/{d}/eval_times.txt", "a") as file:
            file.write(f"{datetime.datetime.now()} {d}: {str(eval_time)} \n")

        linear.utils.save_pipeline_threshold_experiment(
            f"runs/iter-thresh-retrain/{threshold_method}/{d}/models",
            preprocessor,
            model,
            filename=f"iter-thresh-retrain.pickle",
        )

        libmultilabel.common_utils.dump_log(
            metrics=metrics_collection,
            split="test",
            log_path=f"runs/iter-thresh-retrain/{threshold_method}/{d}/logs/iter-thresh-retrain.json",
        )
