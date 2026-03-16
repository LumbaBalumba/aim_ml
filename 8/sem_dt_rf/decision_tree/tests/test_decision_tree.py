import numpy as np
import unittest

from sem_dt_rf.decision_tree.decision_tree import ClassificationDecisionTree


class TestDecisionTree(unittest.TestCase):
    def test_small_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(10, 2)),
            np.random.normal(loc=(-5, 5), size=(10, 2)),
            np.random.normal(loc=(5, -5), size=(10, 2)),
            np.random.normal(loc=(5, 5), size=(10, 2)),
        ))
        y = np.array(
            [0] * 20 + [1] * 20
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1

    def test_decision_tree(self):
        np.random.seed(1)
        clas_tree = ClassificationDecisionTree(max_depth=4, min_leaf_size=1)
        x = np.vstack((
            np.random.normal(loc=(-5, -5), size=(100, 2)),
            np.random.normal(loc=(-5, 5), size=(100, 2)),
            np.random.normal(loc=(5, -5), size=(100, 2)),
            np.random.normal(loc=(5, 5), size=(100, 2)),
        ))
        y = np.array(
            [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100
        )
        clas_tree.fit(x, y)
        predictions = clas_tree.predict(x)
        assert (predictions == y).mean() == 1

