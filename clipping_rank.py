"""
Usage:
    clipping_rank.py [<checkpoint_dir>...] [-o output] [-m mode]

Options:
    -o output       output path [default: results.json]
    -m mode         computation mode [default: np]
"""

from collections import defaultdict
import json
from libmultilabel import linear
import docopt
import numpy as np
import sklearn.decomposition
import scipy
import scipy.sparse as sparse
import re


def np_clip(weights, absweights, eps):
    clipped = absweights < eps
    weights[clipped] = 0
    return weights, absweights


def sparse_clip(weights, absweights, eps):
    if not isinstance(weights, sparse.csr_matrix):
        weights = sparse.csr_matrix(weights)
        absweights = sparse.csr_matrix(absweights)

    clipped = absweights.data < eps
    weights.data[clipped] = 0
    weights.eliminate_zeros()
    return weights, np.abs(weights)


def sklearn_svd(weights, cutoffs):
    pca = sklearn.decomposition.PCA(
        copy=True,
        svd_solver="full",
        random_state=np.random.randint(2**31 - 1),
    )
    pca.fit(weights)
    s = pca.singular_values_
    r = [int(np.sum(pca.explained_variance_ratio_ > cutoff)) for cutoff in cutoffs]
    return s.tolist(), r


def np_svd(weights, cutoffs):
    s = np.linalg.svd(weights, compute_uv=False)
    s2 = s**2
    explained_variance_ratio = s2 / s2.sum()
    r = [int(np.sum(explained_variance_ratio > cutoff)) for cutoff in cutoffs]
    return s.tolist(), r


def scipy_svd(weights, cutoffs):
    s = scipy.linalg.svd(weights, compute_uv=False)
    s2 = s**2
    explained_variance_ratio = s2 / s2.sum()
    r = [int(np.sum(explained_variance_ratio > cutoff)) for cutoff in cutoffs]
    return s.tolist(), r


def main():
    args = docopt.docopt(__doc__)
    np.random.seed(42)

    linear_qs = 100
    geo_qs = 40
    geo_start = 0.8
    geo_step = 0.8
    quantiles = np.hstack(
        [
            np.arange(0, linear_qs) / linear_qs,
            1 - (1 - geo_start) * geo_step ** np.arange(geo_qs),
        ]
    )

    results = {}
    for d in args["<checkpoint_dir>"]:
        run_name = d.split("/")[-2]
        m = re.match(r"([^_]+)_([^_]+_[^_]+)_c(\d+)_.*", run_name)
        data = m[1]
        solver = m[2]
        reg = m[3]

        preprocessor, model = linear.load_pipeline(d)
        weights = model["weights"].A
        absweights = np.abs(weights)
        epss = np.quantile(absweights, quantiles)

        cutoffs = 0.01 * 0.1 ** np.arange(3)
        ranks = {}
        singular_values = {}
        for eps in epss:
            if args["-m"] == "np":
                weights, absweights = np_clip(weights, absweights, eps)
                s, r = np_svd(weights, cutoffs)
            elif args["-m"] == "sp":
                weights, absweights = sparse_clip(weights, absweights, eps)
                s, r = scipy_svd(weights, cutoffs)
            elif args["-m"] == "sklearn":
                weights, absweights = np_clip(weights, absweights, eps)
                s, r = sklearn_svd(weights, cutoffs)

            singular_values[eps] = s
            ranks[eps] = r

        results[run_name] = {
            "data": data,
            "solver": solver,
            "reg": reg,
            "max_rank": min(weights.shape[0], weights.shape[1]),
            "cutoffs": cutoffs.tolist(),
            "ranks": ranks,
            "singular_values": singular_values,
        }

    with open(args["-o"], "w") as f:
        json.dump(
            {
                "quantiles": {
                    "linear_qs": linear_qs,
                    "geo_qs": geo_qs,
                    "geo_start": geo_start,
                    "geo_step": geo_step,
                },
                "results": results,
            },
            f,
        )


if __name__ == "__main__":
    main()
