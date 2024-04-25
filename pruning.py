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
    evaluate: typing.Callable[[float, sparse.csr_matrix | sparse.csc_matrix], npt.NDArray],
    evaluate_shape: tuple,
):
    argsort_ts = np.argsort(ts)
    if isinstance(evaluate_shape, int):
        evaluate_shape = (evaluate_shape,)
    if evaluate_shape[0] != 0:
        result = np.zeros((ts.size, *evaluate_shape))

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
        if evaluate_shape[0] != 0:
            result[argsort_ts[i]] = evaluate(t, pruned_weights)
        else:
            evaluate(t, pruned_weights)

    if evaluate_shape[0] != 0:
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
) -> dict[str, dict[str, float]]:
    num_instance = x.shape[0]
    results = {}

    def evaluate(t: float, pruned_weights: sparse.csr_matrix | sparse.csc_matrix):
        metric_collection = linear.get_metrics(metrics, y.shape[1])
        with _terrible_python_language_design(model, pruned_weights):
            for i in tqdm.tqdm(range(math.ceil(num_instance / eval_batch_size))):
                slice = np.s_[i * eval_batch_size : (i + 1) * eval_batch_size]
                preds = linear.predict_values(model, x[slice])
                target = y[slice].toarray()
                metric_collection.update(preds, target)
        results[t] = metric_collection.compute()

    approx(model, ts, evaluate, 0)
    return results
