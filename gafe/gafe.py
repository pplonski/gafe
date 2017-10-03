import json
import time
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score

from evaluator import Evaluator
from population import Population

class GAFE:

    def __init__(self, evaluator = Evaluator(), duration = 5, new_features_lower_cnt = 2, new_features_upper_cnt = 10):
        self._duration = duration # in minutes
        self._population_count = 10
        self._new_features_lower_cnt = new_features_lower_cnt
        self._new_features_upper_cnt = new_features_upper_cnt
        self._population = None
        self._evaluator = evaluator
        self._best = None
        self._base_score = None

    def _get_base_score_and_assess_parameters(self, X, y, X_vald = None, y_vald = None):
        time_start = time.time()
        self._base_score = self._evaluator.evaluate(X, y, X_vald, y_vald)
        print('Base score', self._base_score)
        time_to_compute_base = time.time() - time_start
        time_to_compute_base *= (X.shape[1]+(self._new_features_upper_cnt - self._new_features_lower_cnt)/2.0)/float(X.shape[1]) # scale

        self._population_count = max(10, int(self._duration*60.0/time_to_compute_base/6.0))
        print('Population', self._population_count)


    def fit(self, X, y, X_vald = None, y_vald = None):
        self._get_base_score_and_assess_parameters(X, y, X_vald, y_vald)
        self._population = Population(self._evaluator, self._population_count, X.shape[1], self._new_features_lower_cnt, self._new_features_upper_cnt)
        self._population.random_init()

        end_time = datetime.now() + timedelta(minutes=self._duration)
        cnt = 0
        while datetime.now() < end_time or cnt < 3:
            _ = self._population.evaluate(X, y, X_vald, y_vald)
            self._population.evolve()
            best_nfs, best_score = self._population.best()
            improvement = np.round((self._base_score - best_score) / self._base_score * 100.0, 5)
            print("Iteration: {0}, score: {1}, improvement: {2}%".format(cnt, best_score, improvement))
            cnt += 1

    def best(self):
        return self._population.best()
