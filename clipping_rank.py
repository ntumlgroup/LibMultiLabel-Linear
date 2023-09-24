"""
Usage:
    clipping_rank.py [<checkpoint_dir>...] [-o output]

Options:
    -o output       output path [default: results.json]
"""

from collections import defaultdict
import json
from libmultilabel import linear
import docopt
import numpy as np
import sklearn.decomposition
import re


def clip(weights, absweights, eps):
    clipped = absweights < eps
    weights[clipped] = 0
    return weights


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
            pca = sklearn.decomposition.PCA(
                copy=True,
                svd_solver="full",
                random_state=np.random.randint(2**31 - 1),
            )
            pca.fit(clip(weights, absweights, eps))
            singular_values[eps] = pca.singular_values_.tolist()
            ranks[eps] = [
                int(np.sum(pca.explained_variance_ratio_ > cutoff))
                for cutoff in cutoffs
            ]

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
