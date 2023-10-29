from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from . import linear
from . import tree

class EnsembleModel:
    def __init__(self,
                 models : list[tree.TreeModel],
    ) -> None:
        self.models = models
        self.name = "ensemble"

    def predict_values(self,
                       x : sparse.csr_matrix,
                       beam_width : int = 10,
    ) -> np.ndarray:
        """Calculates the decision values associated with x.
        
        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Default to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        all_preds = [linear.predict_values(self.models[i], x) for i in range(3)]
        return (np.median(all_preds, axis=0) + np.max(all_preds, axis=0)) / 2
    
def train_ensemble(
    y: sparse.csr_matrix,
    x: sparse.csr_matrix,
    options: str = "",
    K=100,
    dmax=10,
    verbose: bool = True,
) -> EnsembleModel: 
    """Trains an ensemble model using linear tree model as base model

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


    return EnsembleModel([tree.train_tree(y, x, options, K, dmax, verbose) for i in range(3)])
