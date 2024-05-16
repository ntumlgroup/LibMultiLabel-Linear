import numpy as np
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import pruning



if __name__ == "__main__":
    name = "lexglue/ecthr_a"
    datapath = f"data/{name}"
    preprocessor = linear.Preprocessor()
    dataset = linear.load_dataset("txt", f"{datapath}/train.txt", f"{datapath}/test.txt")
    dataset = preprocessor.fit_transform(dataset)

    model = pruning.iter_thresh(dataset, quantile = .20 ,  quantile_multiple = 1.05, perf_drop_tolerance = .01, K=100, dmax=10)