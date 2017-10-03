'''
Evaluator tests.
'''
import os
import unittest
import numpy as np
from sklearn.datasets import load_iris

from gafe.evaluator import Evaluator

class EvaluatorTest(unittest.TestCase):

    def test_init(self):
        eva = Evaluator()
        self.assertEqual(eva._metric, 'neg_log_loss')
        self.assertEqual(eva._cv_folds, 5)

    def test_evolve(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        eva = Evaluator()
        score = eva.evaluate(X, y)
        self.assertTrue(score < 0.25)
        score = eva.evaluate(X, y, X_vald=X, y_vald=y)
        self.assertTrue(score < 0.25)
