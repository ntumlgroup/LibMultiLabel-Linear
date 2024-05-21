from __future__ import annotations
import contextlib
import math

import tqdm

import libmultilabel.linear as linear
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import typing


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


def iter_thresh(
    dataset,
    quantile: float,
    quantile_multiple: float,
    perf_drop_tolerance=0.01,
    K=100,
    dmax=10,
) -> linear.tree.TreeModel:
    import copy

    root = linear.tree._tree_cache("tree_cache", dataset["train"]["y"], dataset["train"]["x"], K, dmax)
    model_a = linear.tree.train_tree_thresh(dataset["train"]["y"], dataset["train"]["x"], root, quantile)

    model_b = copy.deepcopy(model_a)
    model_b.flat_model.weights = linear.utils.threshold_by_label(
        model_b.flat_model.weights.tocsc(), quantile * quantile_multiple
    )

    metric_a = iter_thresh_evaluate(model_a, dataset, ["P@1"])
    metric_b = iter_thresh_evaluate(model_b, dataset, ["P@1"])

    q = quantile
    k = 1
    if np.abs(metric_a["P@1"] - metric_b["P@1"]) > perf_drop_tolerance:
        while np.abs(metric_a["P@1"] - metric_b["P@1"]) > perf_drop_tolerance:
            model_a = linear.tree.train_tree_thresh(dataset["train"]["y"], dataset["train"]["x"], root, q)
            q = q / quantile_multiple
            model_b = linear.tree.train_tree_thresh(dataset["train"]["y"], dataset["train"]["x"], root, q)
            metric_a = iter_thresh_evaluate(model_a, dataset, ["P@1"])
            metric_b = iter_thresh_evaluate(model_b, dataset, ["P@1"])
            k += 1
        return model_a
    else:
        model_k_a, model_k_b = model_a, model_b
        metric_k_a, metric_k_b = metric_a, metric_b
        while np.abs(metric_k_a["P@1"] - metric_k_b["P@1"]) < perf_drop_tolerance:
            metric_k_a, model_k_a = metric_k_b, model_k_b
            model_k_b.flat_model.weights = linear.utils.threshold_by_label(model_k_b.flat_model.weights.tocsc(), q)
            metric_k_b = iter_thresh_evaluate(model_k_b, dataset, ["P@1"])
            q *= quantile_multiple
            k += 1
        return model_k_a


def iter_thresh_evaluate(
    model,
    dataset,
    metrics: list[str],
    eval_batch_size: int = 256,
) -> dict[str, dict[str, float]]:

    num_instance = dataset["test"]["x"].shape[0]
    results = {}

    metric_collection = linear.get_metrics(metrics, dataset["test"]["y"].shape[1])
    for i in tqdm.tqdm(range(math.ceil(num_instance / eval_batch_size))):
        slice = np.s_[i * eval_batch_size : (i + 1) * eval_batch_size]
        preds = linear.predict_values(model, dataset["test"]["x"][slice])
        target = dataset["test"]["y"][slice].toarray()
        metric_collection.update(preds, target)

    results = metric_collection.compute()
    return results
