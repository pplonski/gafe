import numpy as np
from new_feature_set import NewFeatureSet

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier


class Population:

    def __init__(self, evaluator, count, original_features_cnt, new_features_lower_cnt, new_features_upper_cnt):
        self._evaluator = evaluator
        self._count = count
        self._original_features_cnt = original_features_cnt
        self._new_features_lower_cnt = new_features_lower_cnt
        self._new_features_upper_cnt = new_features_upper_cnt
        self._sets = {}
        self._score_book = {} # log all results
        # checks
        if (self._new_features_upper_cnt < self._new_features_lower_cnt) or \
            (self._new_features_lower_cnt < 1):
            raise ValueError('New features count wrong values')

    def random_init(self):
        while len(self._sets) < self._count:
            new_features_cnt = np.random.random_integers(self._new_features_lower_cnt, self._new_features_upper_cnt)
            nfs = NewFeatureSet()
            nfs.random_init(self._original_features_cnt, new_features_cnt)
            k = nfs.key()
            if k not in self._sets:
                self._sets[k] = {'nfs': nfs, 'score': None}

    def evaluate(self, X, y, X_vald = None, y_vald = None):
        scores = []
        for key, d in self._sets.iteritems():
            nfs = d['nfs']
            if key not in self._score_book:
                new_X = nfs.transform(X)
                new_X_vald = None if X_vald is None else nfs.transform(X_vald)
                score = self._evaluator.evaluate(new_X, y, new_X_vald, y_vald)
                self._score_book[key] = {'nfs': nfs, 'score': score}
                self._sets[key]['score'] = score
            else:
                # Already in score book
                self._sets[key]['score'] = self._score_book[key]['score']
            scores += [self._sets[key]['score']]
        return scores

    def evolve(self):
        scores = sorted([i['score'] for k,i in self._score_book.iteritems()])
        self._sets.clear()
        scores = sorted([i['score'] for k,i in self._score_book.iteritems()])
        l = sorted(self._score_book.items(), key=lambda x: x[1]['score'], reverse=True)
        for i in l[:self._count]:
            nfs = i[1]['nfs']
            nfs.mutate(self._original_features_cnt, mutate_features_cnt = 1)
            self._sets[nfs.key()] = {'nfs': nfs, 'score': None}

    def best(self):
        scores = sorted([i['score'] for k,i in self._score_book.iteritems()])
        best_score = scores[-1]
        if best_score is None:
            return None, None
        for k, v in self._score_book.iteritems():
            if v['score'] == best_score:
                return v['nfs'], v['score']
        return None, None
