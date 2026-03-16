import unittest
import numpy as np

from sem_dt_rf.decision_tree.criterio import GiniCriterion, EntropyCriterion


class TestGiniCriterion(unittest.TestCase):
    def test_gini_get_predict_val(self):
        target = np.array([1, 1, 2, 4, 2, 2, 0, 1, 0, 4])
        y_true = np.array([0.2, 0.3, 0.3, 0, 0.2])
        y_pred = GiniCriterion(n_classes=5).get_predict_val(target)
        assert np.allclose(y_pred, y_true)

    def test_gini_score(self):
        target = np.array([1, 1, 2, 4, 2, 2, 0, 1, 0, 4])
        scores = GiniCriterion(n_classes=5).score(target)
        assert np.isclose(scores, 0.74)

    def test_gini_get_best_split(self):
        n = 100
        x = np.arange(n)
        num_zeros = n // 2
        y = np.r_[np.ones(num_zeros), np.zeros(n - num_zeros)].astype(int)
        threshold, q_best = GiniCriterion(n_classes=2).get_best_split(x, y)
        assert np.isclose(threshold, 49.5)
        assert np.isclose(q_best, 0.5)

    def test_gini_get_best_split_2(self):
        x = np.array([0.1, 0.4, 0.2, 0.3, 0.5, 0])
        y = np.array([0, 1, 0, 1, 1, 0])
        threshold, _ = GiniCriterion(n_classes=2).get_best_split(x, y)
        assert np.isclose(threshold, 0.25)


class TestEntropyCriterion(unittest.TestCase):
    def test_entropy_get_predict_val(self):
        target = np.array([1, 1, 2, 4, 2, 2, 0, 1, 0, 4])
        y_true = np.array([0.2, 0.3, 0.3, 0, 0.2])
        y_pred = EntropyCriterion(n_classes=5).get_predict_val(target)
        assert np.allclose(y_pred, y_true)

    def test_entropy_score(self):
        target = np.array([1, 1, 2, 4, 2, 2, 0, 1, 0, 4])
        scores = EntropyCriterion(n_classes=5).score(target)
        assert np.isclose(scores, 1.3661548)

    def test_entropy_get_best_split(self):
        n = 100
        x = np.arange(n)
        num_zeros = n // 2
        y = np.r_[np.ones(num_zeros), np.zeros(n - num_zeros)].astype(int)
        threshold, q_best = EntropyCriterion(n_classes=2).get_best_split(x, y)
        assert np.isclose(threshold, 49.5)
        assert np.isclose(q_best, 0.69314618)

    def test_entropy_get_best_split_v2(self):
        x = np.array([0.1, 0.4, 0.2, 0.3, 0.5, 0])
        y = np.array([0, 1, 0, 1, 1, 0])
        threshold, _ = EntropyCriterion(n_classes=2).get_best_split(x, y)
        assert np.isclose(threshold, 0.25)

