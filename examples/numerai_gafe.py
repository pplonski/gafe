from __future__ import print_function
import pandas as pd
import numpy as np
from gafe import GAFE
from gafe.new_feature_set import NewFeatureSet
np.random.seed(123) # your custom seed

# load numerai data
train = pd.read_csv('~/sandbox/numerai/numerai_training_data.csv')
test  = pd.read_csv('~/sandbox/numerai/numerai_tournament_data.csv')
vald = test[test.data_type == 'validation'] # get validation samples

# train data
X = np.asarray(train[[i for i in train.columns if 'feature' in i]])
y = train.target
# validation data
X_vald = np.asarray(vald[[i for i in vald.columns if 'feature' in i]])
y_vald = vald.target
# validation + test data
X_test = np.asarray(test[[i for i in test.columns if 'feature' in i]])
y_test = test.target

# search for new features, number of new features from 10 to 30, search for about 1 minute (the time run is estimated)
gafe = GAFE(duration = 1, new_features_lower_cnt = 10, new_features_upper_cnt = 30)
gafe.fit(X, y, X_vald, y_vald) # use train and validation data

nfs, score = gafe.best()
print('Score after gafe:', score)
print('New features:', nfs.name([i for i in train.columns if 'feature' in i]))

nfs.save('./numerai_new_features.json') # save new features transformation in json

# transform and save data
X_new = nfs.transform(X) # new train set
X_train_new = nfs.transform(X_test) # new test set

cols = [i for i in train.columns if 'feature' in i]
cols += nfs.name([i for i in train.columns if 'feature' in i])


df = pd.DataFrame(data=X_new, columns = cols)
df['target'] = y
df.to_csv('./numerai_train_gafe.csv', index=False)

df = pd.DataFrame(data=X_test_new, columns = cols)
df['target'] = y_test
df['id'] = test['id']
df.to_csv('./numerai_test_gafe.csv', index=False)
