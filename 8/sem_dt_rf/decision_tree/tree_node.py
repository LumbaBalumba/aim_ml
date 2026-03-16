from typing import Tuple
import numpy as np

from sem_dt_rf.decision_tree.criterio import Criterion


class TreeNode:
    def __init__(self, depth: int):
        self.depth = depth

        self.threshold = None
        self.feature_id = None
        self.q_value_max = None

        self.left_child = None
        self.right_child = None

        self.predictions = None

    def is_terminal(self) -> bool:
        return self.left_child is None and self.right_child is None

    def set_predictions(self, predictions: np.ndarray):
        self.predictions = predictions

    def find_best_split(self, x: np.ndarray, y: np.ndarray, criterion: Criterion):
        """
        Finds best split for current node

        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)
        y : target values, np.ndarray.shape = (n_samples, )
        criterion: criterion
        """
        tr_best, q_max, feature_id_best  = None, -np.inf, -1
        for i in range(x.shape[1]):
            tr_current, q_current = criterion.get_best_split(x[:, i], y)
            if q_current > q_max:
                q_max = q_current
                feature_id_best = i
                tr_best = tr_current

        self.feature_id = feature_id_best
        self.q_value_max = q_max
        self.threshold = tr_best

    def get_best_split_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : samples in node, np.ndarray.shape = (n_samples, n_features)

        Returns
        -------
        right_mask : indicates samples in right node after split
            np.ndarray.shape = (n_samples, )
            np.ndarray.dtype = bool
        """
        return np.asarray(x[:, self.feature_id] < self.threshold)

    def create_children(self):
        self.left_child = TreeNode(self.depth + 1)
        self.right_child = TreeNode(self.depth + 1)
