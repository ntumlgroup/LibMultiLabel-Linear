import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing

__all__ = ["spherical", "balanced_spherical", "lloyd"]


def spherical(x: sparse.csr_matrix, k: int, max_iter: int, tol: float) -> np.ndarray:
    """Perform spherical k-means clustering on x.

    Args:
        x (sparse.csr_matrix): Matrix with dimensions number of points * dimension of underlying space.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping tolerance. Lower for more optimal clustering but longer run time.

    Returns:
        np.ndarray: An array of cluster indices.
    """
    num_points = x.shape[0]

    x = sklearn.preprocessing.normalize(x, norm="l2", axis=1)
    init = np.random.choice(np.arange(num_points), size=k, replace=False)

    centroids = x[init]  # k * dimension of underlying space
    prev_sim = np.inf
    for _ in range(max_iter):
        similarity = x * centroids.T  # number of points * k
        cluster = similarity.argmax(axis=1).A1

        avg_sim = np.take_along_axis(similarity, cluster.reshape(-1, 1), axis=1).sum() / num_points
        if prev_sim - avg_sim < tol:
            return cluster

        centroids = np.zeros(centroids.shape)
        for i in range(k):
            xcluster = x[cluster == i]
            if xcluster.shape[0] > 0:
                centroids[i] = xcluster.sum(axis=0).A1
            else:
                centroids[i] = x[np.random.choice(np.arange(num_points))].toarray()
        # should centroids be stored as sparse matrices?
        centroids = sparse.csr_matrix(centroids)
        centroids = sklearn.preprocessing.normalize(centroids, norm="l2", axis=1)
        prev_sim = avg_sim

    return cluster


def balanced_spherical(x: sparse.csr_matrix, k: int, max_iter: int, tol: float) -> np.ndarray:
    """Perform balanced spherical k-means clustering on x.

    Args:
        x (sparse.csr_matrix): Matrix with dimensions number of points * dimension of underlying space.
        k (int): Number of clusters. Must be a power of 2.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping tolerance. Lower for more optimal clustering but longer run time.

    Returns:
        np.ndarray: An array of cluster indices.
    """
    if k <= 0 or k & (k - 1) != 0:
        raise ValueError(f"k={k} is not power of two")

    num_points = x.shape[0]

    def cluster_each(indices):
        for index in indices:
            cluster = _balanced_spherical_2means(x[index], max_iter, tol)
            for i in range(2):
                yield index[cluster == i]

    n = k.bit_length() - 1  # k = 2**n
    indices = [np.arange(num_points)]
    for _ in range(n):
        indices = cluster_each(indices)

    cluster = np.empty(num_points, dtype=int)
    for i, index in enumerate(indices):
        cluster[index] = i

    return cluster


def _balanced_spherical_2means(x: sparse.csr_matrix, max_iter: int, tol: float) -> np.ndarray:
    num_points = x.shape[0]
    x = sklearn.preprocessing.normalize(x, norm="l2", axis=1)

    init = np.random.choice(np.arange(num_points), size=2, replace=False)
    centroids = x[init]  # 2 * dimension of underlying space

    prev_sim = np.inf
    for _ in range(max_iter):
        centroid_diff = centroids[1] - centroids[0]
        similarity_diff = (centroid_diff * x.T).toarray()  # 1 * number of points
        similarity_rank = np.argsort(similarity_diff)

        cluster = np.zeros(num_points, dtype=int)
        cluster[similarity_rank[: num_points // 2]] = 1

        avg_sim = similarity_diff * (2 * cluster - 1) / (2 * num_points)
        if prev_sim - avg_sim < tol:
            return cluster

        centroids = np.zeros(centroids.shape)
        for i in range(2):
            xcluster = x[cluster == i]
            if xcluster.shape[0] > 0:
                centroids[i] = xcluster.sum(axis=0).A1
            else:
                centroids[i] = x[np.random.choice(np.arange(num_points))].toarray()
        # should centroids be stored as sparse matrices?
        centroids = sparse.csr_matrix(centroids)
        centroids = sklearn.preprocessing.normalize(centroids, norm="l2", axis=1)
        prev_sim = avg_sim

    return cluster


def lloyd(x: sparse.csr_matrix, k: int, max_iter: int, tol: float) -> np.ndarray:
    """Perform lloyd's algorithm k-means clustering on x.

    Args:
        x (sparse.csr_matrix): Matrix with dimensions number of points * dimension of underlying space.
        k (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        tol (float): Stopping tolerance. Lower for more optimal clustering but longer run time.

    Returns:
        np.ndarray: An array of cluster indices.
    """
    num_points = x.shape[0]
    init = np.random.choice(np.arange(num_points), size=k, replace=False)
    centroids = x[init]  # k * dimension of underlying space
    prev_dist = np.inf
    for _ in range(max_iter):
        similarity = x * centroids.T  # number of points * k
        norm_square = centroids.power(2).sum(axis=1).T  # number of points * k
        distance_square = norm_square - 2 * similarity  # number of points * k
        cluster = distance_square.argmin(axis=1).A1

        avg_dist = np.take_along_axis(distance_square, cluster.reshape(-1, 1), axis=1).sum() / num_points
        if prev_dist - avg_dist < tol:
            return cluster

        centroids = np.zeros(centroids.shape)
        for i in range(k):
            xcluster = x[cluster == i]
            if xcluster.shape[0] > 0:
                centroids[i] = xcluster.mean(axis=0).A1
            else:
                centroids[i] = x[np.random.choice(np.arange(num_points))].toarray()
        # should centroids be stored as sparse matrices?
        centroids = sparse.csr_matrix(centroids)
        prev_dist = avg_dist

    return cluster
