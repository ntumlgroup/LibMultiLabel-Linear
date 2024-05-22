import numpy as np
import libmultilabel.linear as linear
import importlib as fp
import matplotlib.pyplot as plt
import pruning
import time


if __name__ == "__main__":    
    datasets = [
        # "eur-lex",
        # "rcv1",
        # "mimic"
        # "wiki10-31k",
        # "amazoncat-13k/ver1"
        "lexglue/ecthr_a"
    ]
    from sklearn.model_selection import train_test_split
    for d in datasets:
        np.random.seed = 1337
        datapath = f"data/{d}"
        preprocessor = linear.Preprocessor()
        dataset = linear.load_dataset("txt", f"{datapath}/train.txt", f"{datapath}/test.txt")
        dataset = preprocessor.fit_transform(dataset)



        root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"], K=100, dmax=10)
        
        model = linear.tree.train_tree(dataset["train"]["y"], dataset["train"]["x"], "-s 1 -B 1 -e 0.0001 -q")
        metrics = pruning.iter_thresh_evaluate(model, dataset["test"]["y"], dataset["test"]["x"], ["P@1", "P@3", "P@5"])
        results = linear.tabulate_metrics(metrics, "test")
        print(results)

        model = linear.tree.train_tree_thresh(dataset["train"]["y"], dataset["train"]["x"], root, 0, "-s 1 -B 1 -e 0.0001 -q") 
        metrics = pruning.iter_thresh_evaluate(model, dataset["test"]["y"], dataset["test"]["x"], ["P@1", "P@3", "P@5"])
        results = linear.tabulate_metrics(metrics, "test")
        print(results)

        # with open("runs/logs/compare_model_split.txt", "a") as fp:
        #     fp.write(f"Full Training Results {d}: \n nnz: {np.sum(model.flat_model.weights != 0)} \n {results} \n")


        x_tr, x_val, y_tr, y_val = train_test_split(dataset["train"]["x"], dataset["train"]["y"], test_size=.2, train_size=.8, random_state=2)
        model = linear.tree.train_tree_thresh(y_tr, x_tr, root, 0, "-s 1 -B 1 -e 0.0001 -q") 
        metrics = pruning.iter_thresh_evaluate(model, dataset["test"]["y"], dataset["test"]["x"], ["P@1", "P@3", "P@5"])
        results = linear.tabulate_metrics(metrics, "test")
        print(results)
        # with open("runs/logs/compare_model_split.txt", "a") as fp:
        #     fp.write(f"80/20 Training Results {d}: \n nnz: {np.sum(model.flat_model.weights != 0)} \n {results} \n")
