from __future__ import annotations
import contextlib
import math

import tqdm

import libmultilabel.linear as linear
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import typing

full_weights = 0


def approx(
    model,
    ts: list[float],
    callback: typing.Callable[[float, sparse.csr_matrix | sparse.csc_matrix], npt.NDArray],
    callback_return_shape: tuple | int,
):
    argsort_ts = np.argsort(ts)
    if isinstance(callback_return_shape, int):
        callback_return_shape = (callback_return_shape,)
    if callback_return_shape[0] != 0:
        result = np.zeros((ts.size, *callback_return_shape))

    threshold_denom = np.zeros(model.flat_model.weights.shape[1])

    def visit(node):
        if node.label_map.size == 0:
            return
        slice = np.s_[model.weight_map[node.index] : model.weight_map[node.index + 1]]
        threshold_denom[slice] = node.threshold_denom

    model.root.dfs(visit)

    weights = model.flat_model.weights
    matrix = type(weights)
    shape = weights.shape

    data = weights.data
    indices = weights.indices
    indptr = weights.indptr.copy()
    absdata = np.abs(data)
    for i, t in enumerate(ts[argsort_ts]):
        threshold = t / threshold_denom
        kept = absdata >= threshold[indices]
        cumkept = np.cumsum(kept)

        data = data[kept]
        indices = indices[kept]
        indptr[1:] = cumkept[indptr[1:] - 1]
        absdata = absdata[kept]

        pruned_weights = matrix((data, indices, indptr), shape=shape)
        if callback_return_shape[0] != 0:
            result[argsort_ts[i]] = callback(t, pruned_weights)
        else:
            callback(t, pruned_weights)

    if callback_return_shape[0] != 0:
        return result


@contextlib.contextmanager
def _terrible_python_language_design(model, weights):
    original = model.flat_model.weights
    try:
        model.flat_model.weights = weights
        yield "python is terrible"
    finally:
        model.flat_model.weights = original


def approx_evaluate(
    model,
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    metrics: list[str],
    ts: list[float],
    eval_batch_size: int = 256,
    callback: typing.Callable[[float, sparse.csr_matrix | sparse.csc_matrix], typing.Any] = None,
) -> dict[str, dict[str, float]]:
    num_instance = x.shape[0]
    results = {}

    def evaluate(t: float, pruned_weights: sparse.csr_matrix | sparse.csc_matrix):
        metric_collection = linear.get_metrics(metrics, y.shape[1], stable=True)
        with _terrible_python_language_design(model, pruned_weights):
            for i in tqdm.tqdm(range(math.ceil(num_instance / eval_batch_size))):
                slice = np.s_[i * eval_batch_size : (i + 1) * eval_batch_size]
                preds = linear.predict_values(model, x[slice])
                target = y[slice].toarray()
                metric_collection.update(preds, target)
            if callback is not None:
                callback(t, pruned_weights)

        results[t] = metric_collection.compute()

    approx(model, ts, evaluate, 0)
    return results


def concat_thresholds(model, num_quantiles):
    thresholds = np.zeros((num_quantiles, model.flat_model.weights.shape[1]))  # number of quantiles * total labels

    def visit(node):
        if node.label_map.size == 0:
            return
        slice = np.s_[model.weight_map[node.index] : model.weight_map[node.index + 1]]
        thresholds[:, slice] = node.thresholds

    model.root.dfs(visit)

    return thresholds


def iter_thresh(
    name,
    dataset,
    initial_quantile: float,
    perf_drop_tolerance=0.01,
    K=100,
    dmax=10,
    option="",
) -> linear.tree.TreeModel:
    import copy
    import datetime
    from sklearn.model_selection import train_test_split

    if name == "amazoncat-13k/ver1":
        name = "amazoncat-13k"
    with open(f"runs/logs/{name}-iter-thresh-logs.txt", "a") as fp:
        fp.write(f"{datetime.datetime.now()} \n")

    np.random.seed(1337)
    root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"], K=100, dmax=10)
    x_tr, x_val, y_tr, y_val = x_tr, x_val, y_tr, y_val = train_test_split(
        dataset["train"]["x"], dataset["train"]["y"], test_size=0.2, train_size=0.8, random_state=2
    )
    num_quantiles = 100
    quantiles = [(1 - (1 - initial_quantile) * (0.8**k)) for k in range(100)]  # adjust when needed or make input
    model_a = linear.tree.train_tree_compute_threshold(
        y_tr, x_tr, root, quantiles, options=option, with_thresholding=True
    )

    thresholds = concat_thresholds(model_a, num_quantiles)

    k = 0
    model_a.flat_model.weights = model_a.flat_model.weights.tocsc()
    model_b = copy.deepcopy(model_a)
    model_b.flat_model.weights = linear.utils.threshold_by_label(model_b.flat_model.weights, thresholds[k + 1, :])

    metric_a = iter_thresh_evaluate(model_a, y_val, x_val, ["P@1"])
    metric_b = iter_thresh_evaluate(model_b, y_val, x_val, ["P@1"])

    if np.abs(metric_a["P@1"] - metric_b["P@1"]) > perf_drop_tolerance:
        while np.abs(metric_a["P@1"] - metric_b["P@1"]) > perf_drop_tolerance:
            quantiles = [(1 - (1 - initial_quantile) * (0.8**i)) for i in range(k, k - 2, -1)]
            model_a = linear.tree.train_tree_compute_threshold(
                y_tr, x_tr, root, quantiles, option, with_thresholding=True
            )
            thresholds = concat_thresholds(model_a, 2)

            model_b = copy.deepcopy(model_a)
            model_b.flat_model.weights = linear.utils.threshold_by_label(
                model_b.flat_model.weights.tocsc(), thresholds[k + 1, :]
            )

            metric_a = iter_thresh_evaluate(model_a, y_val, x_val, ["P@1"])
            metric_b = iter_thresh_evaluate(model_b, y_val, x_val, ["P@1"])

            with open(f"runs/logs/{name}-iter-thresh-logs.txt", "a") as fp:
                fp.write(
                    f"Metric: {metric_a}, Quantile: {quantiles[0]}, nnz: {model_a.flat_model.weights.nnz}, decrease thresh \n"
                )
            k -= 1
        return model_a, quantiles[0]

    else:
        model_k_a, model_k_b = model_a, model_b
        metric_k_a, metric_k_b = metric_a, metric_b

        while np.abs(metric_k_a["P@1"] - metric_k_b["P@1"]) < perf_drop_tolerance:
            metric_k_a = metric_k_b
            model_k_a = copy.deepcopy(model_k_b)
            model_k_b.flat_model.weights = linear.utils.threshold_by_label(
                model_k_b.flat_model.weights, thresholds[k + 1, :]
            )
            metric_k_b = iter_thresh_evaluate(model_k_b, y_val, x_val, ["P@1"])
            with open(f"runs/logs/{name}-iter-thresh-logs.txt", "a") as fp:
                fp.write(
                    f"{k} Metric: {metric_k_a}, Quantile: {quantiles[k]}, nnz: {model_k_a.flat_model.weights.nnz}, increase thresh \n"
                )
            k += 1
        model_k_a.flat_model.weights = model_k_a.flat_model.weights.tocsr()
        return model_k_a, quantiles[k - 1]


def iter_thresh_global(
    name,
    dataset,
    initial_quantile: float,
    perf_drop_tolerance=0.01,
    K=100,
    dmax=10,
    option="",
):
    import copy
    import datetime
    from sklearn.model_selection import train_test_split

    if name == "amazoncat-13k/ver1":
        name = "amazoncat-13k"
    with open(f"runs/logs/{name}-iter-thresh-logs.txt", "a") as fp:
        fp.write(f"{datetime.datetime.now()} \n")

    np.random.seed(1337)
    root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"], K=100, dmax=10)

    x_tr, x_val, y_tr, y_val = x_tr, x_val, y_tr, y_val = train_test_split(
        dataset["train"]["x"], dataset["train"]["y"], test_size=0.2, train_size=0.8, random_state=2
    )

    num_quantiles = 100
    quantiles = [
        (1 - (1 - initial_quantile) * (0.8**k)) for k in range(num_quantiles)
    ]  # adjust when needed or make input

    model_0 = linear.tree.train_tree_compute_threshold(y_tr, x_tr, root, quantiles, options=option)
    model_0.flat_model.weights = model_0.flat_model.weights.tocsc()
    metric_0 = iter_thresh_evaluate(model_0, y_val, x_val, ["P@1"])
    thresholds = concat_thresholds(model_0, num_quantiles)
    i = 1
    model_i = copy.deepcopy(model_0)
    model_i.flat_model.weights = linear.utils.threshold_by_label(model_i.flat_model.weights, thresholds[i, :])
    metric_i = iter_thresh_evaluate(model_i, y_val, x_val, ["P@1"])

    for i in range(2, num_quantiles):
        if (np.abs(metric_i["P@1"] - metric_0["P@1"]) / metric_0["P@1"]) > perf_drop_tolerance:
            model_i.flat_model.weights = linear.utils.threshold_by_label(
                model_0.flat_model.weights, thresholds[i - 1, :]
            )
            metric_i = iter_thresh_evaluate(model_i, y_val, x_val, ["P@1"])
            return model_i, i
        model_i.flat_model.weights = linear.utils.threshold_by_label(model_i.flat_model.weights, thresholds[i, :])
        metric_i = iter_thresh_evaluate(model_i, y_val, x_val, ["P@1"])

        with open(f"runs/logs/{name}-iter-thresh-global-logs.txt", "a") as fp:
            fp.write(f"Metric: {metric_i}, Quantile: {quantiles[i]}, nnz: {model_i.flat_model.weights.nnz}\n")

    return model_i, i


def iter_thresh_evaluate(
    model,
    y_val,
    x_val,
    metrics: list[str],
    eval_batch_size: int = 256,
) -> dict[str, dict[str, float]]:
    num_instance = x_val.shape[0]
    results = {}

    metric_collection = linear.get_metrics(metrics, y_val.shape[1])
    for i in tqdm.tqdm(range(math.ceil(num_instance / eval_batch_size))):
        slice = np.s_[i * eval_batch_size : (i + 1) * eval_batch_size]
        preds = linear.predict_values(model, x_val[slice])
        target = y_val[slice].toarray()
        metric_collection.update(preds, target)

    results = metric_collection.compute()
    return results
