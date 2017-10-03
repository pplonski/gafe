import numpy as np
from new_feature_set import NewFeatureSet

from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer

class Evaluator:

    def __init__(self, cv_fold = 5):
        self._metric = 'neg_log_loss'
        self._cv_folds = cv_fold

    def evaluate(self, X, y, X_vald = None, y_vald = None):
        clf = RandomForestClassifier(n_estimators=32, max_depth=3, n_jobs=-1) # used as base classifier
        if X_vald is None:
            return cross_val_score(clf, X, y, scoring=self._metric, cv=self._cv_folds, n_jobs=-1).mean()
        else:
            clf.fit(X, y)
            sk = get_scorer(self._metric)
            return sk(clf, X_vald, y_vald)
