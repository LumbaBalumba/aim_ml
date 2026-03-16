from typing import Tuple, Union

import numpy as np


class Criterion:
    def get_best_split(self, feature: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """
        Parameters
        ----------
        feature : feature vector, np.ndarray.shape = (n_samples, )
        target  : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        threshold : value to split feature vector, float
        q_value   : impurity improvement, float
        """
        ind = np.argsort(feature)
        feature_sorted = feature[ind]
        target_sorted = target[ind]
        q_all = self.score(target)

        q_max, i_max = -np.inf, -1
        for i in range(1, len(feature_sorted)):
            q_left = self.score(target_sorted[:i])
            q_right = self.score(target_sorted[i:])
            q_current = q_all
            q_current -= (i + 1) / len(feature_sorted) * q_left
            q_current -= (len(feature_sorted) - i - 1) / len(feature_sorted) * q_right
            if q_current > q_max:
                q_max = q_current
                i_max = i

        threshold = (feature_sorted[i_max] + feature_sorted[i_max - 1]) / 2
        return threshold, q_max


    def score(self, target: np.ndarray) -> float:
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        impurity : float
        """
        raise NotImplementedError

    def get_predict_val(self, target: np.ndarray) -> Union[float, np.ndarray]:
        """
        Parameters
        ----------
        target : target vector, np.ndarray.shape = (n_samples, )

        Returns
        -------
        prediction :
            - classification: probability distribution in node, np.ndarray.shape = (n_classes, )
            - regression: best constant approximation, float
        """
        raise NotImplementedError


class GiniCriterion(Criterion):
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def get_predict_val(self, target: np.ndarray) -> np.ndarray:
        return np.bincount(target, minlength=self.n_classes) / target.shape[0]

    def score(self, target):
        probs = self.get_predict_val(target)
        return 1 - np.sum(probs ** 2)


class EntropyCriterion(Criterion):
    EPS = 1e-6

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def get_predict_val(self, target: np.ndarray) -> np.ndarray:
        return np.bincount(target, minlength=self.n_classes) / target.shape[0]

    def score(self, target):
        probs = self.get_predict_val(target)
        return np.sum(-probs * np.log(probs + self.EPS))


class MSECriterion(Criterion):
    def get_predict_val(self, target):
        raise NotImplementedError("Put your code here")

    def score(self, target):
        raise NotImplementedError("Put your code here")
