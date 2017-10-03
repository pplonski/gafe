import numpy as np
import copy

class NewFeature:

    def __init__(self):
        self._operators = [ np.add,        #0
                            np.multiply,   #1
                            np.subtract]   #2
        self._operators_names = ['+', '*', '-']
        self._operator_index = None
        self._columns = []

    def key(self):
        if self._operator_index is None:
            return None
        return 'op_{0}_cols_{1}'.format(self._operator_index, '_'.join([str(j) for j in self._columns]))

    def random_init(self, original_features_cnt):

        self._operator_index = np.random.choice(len(self._operators), 1)[0]
        # operator on [2, 5] original columns
        columns_cnt = 2
        if original_features_cnt > 5: # if there are more original columns please use more cols
            columns_cnt = np.random.choice([2, 2, 2, 2, 3, 3, 3, 4, 4, 5], 1)[0]

        self._columns = np.random.choice(original_features_cnt, columns_cnt, replace=False).tolist()
        if self._operator_index in [0, 1]: # add and mutliply, sort columns, the order dont have meaning
            self._columns = sorted(self._columns)


    def apply(self, X):
        vfunc = np.vectorize(self._operators[self._operator_index])
        col1 = self._columns[0]
        X_col1 = copy.deepcopy(X[:, col1])
        for i in xrange(len(self._columns)-1):
            col2 = self._columns[i+1]
            X_col1 = np.nan_to_num(vfunc(X_col1, X[:, col2]))
        return X_col1

    def _get_col_name(self, index, original_cols_names = []):
        if original_cols_names is None or len(original_cols_names) == 0:
            n = 'col_{0}'.format(index)
        else:
            n = original_cols_names[index]
        return n

    def name(self, original_cols_names = []):
        if len(self._columns) == 0:
            return ''
        n = self._get_col_name(self._columns[0], original_cols_names)
        for i in xrange(len(self._columns)-1):
            n += '{0}'.format(self._operators_names[self._operator_index])
            n += '{0}'.format(self._get_col_name(self._columns[i+1], original_cols_names))
        return n

    def to_json(self):
        return {'operator_index': self._operator_index,
                'columns': self._columns}

    def from_json(self, json_data):
        self._operator_index = json_data.get('operator_index', None)
        self._columns = json_data.get('columns', [])
