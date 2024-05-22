import numpy as np
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import pruning
import time


if __name__ == "__main__":    
    datasets = [
        "eur-lex",
        "rcv1",
        "mimic"
        # "wiki10-31k",
        # "amazoncat-13k/ver1"
    ]
    from sklearn.model_selection import train_test_split
    for d in datasets:
        print(d)
        start = time.perf_counter()
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("txt", f"{datapath}/train.txt", f"{datapath}/test.txt")
        dataset = preprocessor.fit_transform(dataset)

        
        model = pruning.iter_thresh(dataset, quantile=.80,  quantile_multiple=1.05, perf_drop_tolerance=.01, K=100, dmax=10)
        metrics = pruning.iter_thresh_evaluate(model, dataset["test"]["y"], dataset["test"]["x"], ["P@1", "P@3", "P@5"])
        results = linear.tabulate_metrics(metrics, "test") 

        end = time.perf_counter()
        elapsed = end - start
        
        with open("runs/logs/iter_thresh_logs.txt", "a") as fp:
            fp.write(f"{d} iterative threshold: {elapsed} s \n {results} \n")
            print("saved log")

        


