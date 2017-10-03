import json
import numpy as np

from new_feature import NewFeature

class NewFeatureSet:

    def __init__(self):
        self._features = []
        self._keys = []

    def random_init(self, original_features_cnt, new_features_cnt):
        # generate new features
        safe_counter = 0
        while len(self._features) < new_features_cnt and safe_counter < 1000:
            nf = NewFeature()
            nf.random_init(original_features_cnt)
            if nf.key() not in self._keys: # uniue keys = unique new features
                self._features += [nf]
                self._keys += [nf.key()]
            safe_counter += 1

    def key(self):
        if self._keys is None or len(self._keys) == 0:
            return None
        sk = sorted(self._keys)
        return '_'.join(sk)

    def apply(self, X):
        z = np.zeros((X.shape[0], len(self._features)))
        for i, nf in enumerate(self._features):
            z[:, i] = nf.apply(X)
        return z

    def transform(self, X):
        z = self.apply(X)
        return np.concatenate((X, z), axis=1)

    def name(self, original_cols_names = []):
        return [nf.name(original_cols_names) for nf in self._features]

    def mutate(self, original_features_cnt, mutate_features_cnt = 1):
        cnt = 0
        while cnt < mutate_features_cnt:
            mutate_ind = np.random.choice(len(self._features))
            nf = NewFeature()
            nf.random_init(original_features_cnt)
            if nf.key() not in self._keys: # uniue keys = unique new features
                self._features[mutate_ind] = nf
                self._keys[mutate_ind] = nf.key()
                cnt += 1

    def to_json(self):
        data = {}
        for i in xrange(len(self._keys)):
            data[self._keys[i]] = self._features[i].to_json()
        return data

    def from_json(self, json_data):
        self._features, self._keys = [], []
        for k, v in json_data.iteritems():
            nf = NewFeature()
            nf.from_json(v)
            self._features += [nf]
            self._keys += [nf.key()]

    def save(self, filename):
        with open(filename, 'w') as fout:
            fout.write(json.dumps(self.to_json(), indent=4))

    def load(self, filename):
        with open(filename, 'r') as fin:
            data = json.load(fin)
            self.from_json(data)
