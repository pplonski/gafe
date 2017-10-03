'''
Population tests.
'''
import os
import unittest
import numpy as np
from sklearn.datasets import load_iris

from gafe.evaluator import Evaluator
from gafe.population import Population

class PopulationTest(unittest.TestCase):

    def test_init(self):
        pop = Population(Evaluator(), 10, 10, 1, 3)
        self.assertTrue(pop is not None)
        self.assertEqual(len(pop._score_book), 0)


    def test_random_init(self):
        count = 10
        pop = Population(Evaluator(), count, 10, 1, 3)
        pop.random_init()
        self.assertEqual(len(pop._sets), count)

    def test_evaluate_and_evolve(self):
        iris = load_iris() # get data
        X = iris.data
        y = iris.target
        count = 4
        pop = Population(Evaluator(), count, X.shape[1], 1, 3)
        pop.random_init()
        scores = pop.evaluate(X, y)
        self.assertEqual(len(scores), count)
        self.assertEqual(len(pop._score_book), count)
        pop.evolve()
        self.assertEqual(len(pop._sets), count)
