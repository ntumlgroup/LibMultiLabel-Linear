from __future__ import annotations

import hashlib
import logging
import pathlib
import pickle
from typing import Callable

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing
from tqdm import tqdm

from . import linear

__all__ = ["train_tree", "train_tree_approx_pruning", "train_tree_thresh"]


class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None]):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    def __init__(
        self,
        root: Node,
        flat_model: linear.FlatModel,
        weight_map: np.ndarray,
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.weight_map = weight_map
        self.multiclass = False

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
    ) -> np.ndarray:
        """Calculates the decision values associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        # number of instances * number of labels + total number of metalabels
        all_preds = linear.predict_values(self.flat_model, x)
        return np.vstack([self._beam_search(all_preds[i], beam_width) for i in range(all_preds.shape[0])])

    def _beam_search(self, instance_preds: np.ndarray, beam_width: int) -> np.ndarray:
        """Predict with beam search using cached decision values for a single instance.

        Args:
            instance_preds (np.ndarray): A vector of cached decision values of each node, has dimension number of labels + total number of metalabels.
            beam_width (int): Number of candidates considered.

        Returns:
            np.ndarray: A vector with dimension number of classes.
        """
        cur_level = [(self.root, 0.0)]  # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score - np.maximum(0, 1 - pred) ** 2
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        scores = np.full(num_labels, -np.inf)
        for node, score in cur_level:
            slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score - np.maximum(0, 1 - pred) ** 2)
        return scores


def train_tree(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> TreeModel:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    root = _tree_cache("tree_cache", y, x, K, dmax)

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map)

def train_tree_thresh(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    root: Node,
    quantile: float,
    verbose: bool = True,
    options: str = "",
) -> TreeModel:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)
    import time
    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node_threshold(y[relevant_instances], x[relevant_instances], options, node, quantile)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map)


def _tree_cache(
    cache_root: str,
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    K,
    dmax,
) -> Node:
    front = np.s_[:100]
    back = np.s_[100:]
    fingerprint = (
        y.data[front],
        y.indices[front],
        y.data[back],
        y.indices[back],
        y.shape,
        x.data[front],
        x.indices[front],
        x.data[back],
        x.indices[back],
        x.shape,
        K,
        dmax,
        np.random.get_state(),  # used in clustering initializations
        "spherical",  # future-proof clustering method
    )
    h = hashlib.sha256()
    h.update(str(fingerprint).encode())
    digest = h.hexdigest()[:32]
    path = pathlib.Path(f"{cache_root}/{digest}.pickle")
    if path.exists():
        logging.info(f"using tree cache {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        logging.info(f"creating tree cache {path}")
        label_representation = (y.T * x).tocsr()
        label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
        root = _build_tree(label_representation, np.arange(y.shape[1]), 0, K, dmax)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(root, f)
        return root


def train_tree_approx_pruning(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    t: float,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> TreeModel:
    """Trains a linear model for multiabel data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.
    Weight pruning is done according to an approximate upper bound.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        t (float): A proportional constant for weight pruning threshold.
        options (str): The option string passed to liblinear.
        K (int, optional): Maximum degree of nodes in the tree. Defaults to 100.
        dmax (int, optional): Maximum depth of the tree. Defaults to 10.
        verbose (bool, optional): Output extra progress information. Defaults to True.

    Returns:
        A model which can be used in predict_values.
    """
    root = _tree_cache("tree_cache", y, x, K, dmax)

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        _train_node_approx_pruning(
            y[relevant_instances],
            x[relevant_instances],
            t,
            options,
            node,
        )
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map)


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Builds the tree recursively by kmeans clustering.

    Args:
        label_representation (sparse.csr_matrix): A matrix with dimensions number of classes under this node * number of features.
        label_map (np.ndarray): Maps 0..label_representation.shape[0] to the original label indices.
        d (int): Current depth.
        K (int): Maximum degree of nodes in the tree.
        dmax (int): Maximum depth of the tree.

    Returns:
        Node: root of the (sub)tree built from label_representation.
    """
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    metalabels = (
        sklearn.cluster.KMeans(
            K,
            random_state=np.random.randint(2**31 - 1),
            n_init=1,
            max_iter=300,
            tol=0.0001,
            algorithm="elkan",
        )
        .fit(label_representation)
        .labels_
    )

    children = []
    for i in range(K):
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]
        child = _build_tree(child_representation, child_map, d + 1, K, dmax)
        children.append(child)

    return Node(label_map=label_map, children=children)


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    node.model.weights = sparse.csc_matrix(node.model.weights)

def _train_node_threshold(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node, quantile: float):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
        threshold (float): threshold value to prune node weights
    """

    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    node.model.weights = sparse.csc_matrix(node.model.weights)

    weights = node.model.weights
    for i in tqdm(range(node.model.weights.shape[1])):
        col_start, col_end = weights.indptr[i], weights.indptr[i + 1]
        abs_weights = np.abs(weights.data[col_start: col_end])
        threshold = np.quantile(abs_weights, quantile)
        node.model.weights.data[col_start: col_end][abs_weights < threshold] = 0

    node.model.weights.eliminate_zeros()

def _train_node_approx_pruning(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    t: float,
    options: str,
    node: Node,
):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.
    After training, weight pruning is done according to an approximate upper bound.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    if x.shape[0] > 0:
        # edge case with clustering
        quantiles = np.hstack([np.linspace(0, 0.05, 20), np.linspace(0.975, 1, 10)])
        node.wTx_quantiles = np.quantile(linear.predict_values(node.model, x), quantiles, axis=0)
        node.threshold_denom = np.abs(node.wTx_quantiles[0] - 2) * np.sqrt(x.shape[1])
        # threshold = t / node.threshold_denom
        # pruned = np.abs(node.model.weights) < threshold
        # node.model.weights[pruned] = 0

    node.model.weights = sparse.csc_matrix(node.model.weights)


def _flatten_model(root: Node) -> tuple[linear.FlatModel, np.ndarray]:
    """Flattens tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
    Consecutive values of the returned map denotes the start and end indices of the
    weights of each node. Conceptually, given root and node:
        flat_model, weight_map = _flatten_model(root)
        slice = np.s_[weight_map[node.index]:
                      weight_map[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    index = 0
    weights = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        nonlocal index
        node.index = index
        index += 1
        weights.append(node.model.__dict__.pop("weights"))

    root.dfs(visit)

    model = linear.FlatModel(
        name="flattened-tree",
        weights=sparse.hstack(weights, "csr"),
        bias=bias,
        thresholds=0,
        multiclass=False,
    )

    # w.shape[1] is the number of labels/metalabels of each node
    weight_map = np.cumsum([0] + list(map(lambda w: w.shape[1], weights)))

    return model, weight_map
