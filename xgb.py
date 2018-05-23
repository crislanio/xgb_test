from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pickle import load, dump
from os.path import isfile
from gc import collect
from time import time
import xgboost as xgb
import numpy as np

FEATURES_PCT = 0.2
use_gpu = False

id_col = 'SK_ID_CURR'
target_col = 'TARGET'

train = load(open('intermediary/train.pkl', 'rb'))

if not isfile('output/fscores.pkl'):
    fscores = {}
    for column in train.drop(columns=[id_col, target_col]).columns:
        fscores[column] = [1]
    dump(fscores, open('output/fscores.pkl', 'wb'))
else:
    fscores = load(open('output/fscores.pkl', 'rb'))

n_features = len(fscores)
features, probabilities = [feature for feature in fscores],  np.array([fscores[feature][-1] for feature in fscores])
probabilities = probabilities/probabilities.sum()

n_selected_features = round(FEATURES_PCT*n_features)
selected_features = list(np.random.choice(features, size=n_selected_features, replace=False, p=probabilities))

train = train[selected_features+[target_col]].copy()
collect()

X = train.drop(columns=[target_col])
y = train[target_col]

unbalance_factor = train[train[target_col]==0].shape[0]/train[train[target_col]==1].shape[0]
del train
collect()

# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
xgb_params = {
    'eta': 0.25,
    'max_leaves': 2048,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'colsample_bylevel':0.7,
    'min_child_weight':0,
    'alpha': 2,
    'max_depth': 0,
    'scale_pos_weight': unbalance_factor,
    'eval_metric': 'auc',
    'random_state': int(time()),
    'silent': True,
    'grow_policy': 'lossguide',
    'tree_method': 'hist',
    'predictor': 'cpu_predictor',
    'objective': 'binary:logistic'
}

if use_gpu:
    xgb_params.update({'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'objective':'gpu:binary:logistic'})

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1585, random_state=int(time()))
del X, y
collect()

dtrain = xgb.DMatrix(X_train, y_train)
del X_train, y_train
collect()

dvalid = xgb.DMatrix(X_valid, y_valid)

watchlist = [(dvalid, 'validation')]

model = xgb.train(xgb_params, dtrain, 10, watchlist, early_stopping_rounds = 15, verbose_eval=False)
fscores_model = model.get_fscore()
score = roc_auc_score(y_valid, model.predict(xgb.DMatrix(X_valid)))

del X_valid, y_valid
collect()

fscores = load(open('output/fscores.pkl', 'rb'))
for feature in fscores_model:
    fscores[feature].append(fscores[feature][-1] + np.log(fscores_model[feature]))
dump(fscores, open('output/fscores.pkl', 'wb'))

del dtrain, dvalid
collect()

test = load(open('intermediary/test.pkl', 'rb'))[ [id_col]+selected_features ]
test[target_col] = model.predict(xgb.DMatrix(test.drop(columns=[id_col])))

test[[id_col, target_col]].to_csv('output/pre_sub/{}_{}.csv'.format( int(time()), str(score).split('.')[1] ), index=False)
del test
collect()
