from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pickle import load, dump
from os.path import isfile
from gc import collect
from time import time
import xgboost as xgb
import pandas as pd
import numpy as np

FEATURES_PCT = 0.5
SCORE_FEATURES_PCT = 0.8
use_gpu = False

id_col = 'SK_ID_CURR'
target_col = 'TARGET'

train_pkl = load(open('intermediary/train.pkl', 'rb'))

old_tuple = None

while True:
    if not isfile('output/fscores.pkl'):
        fscores = {}
        for column in train_pkl.drop(columns=[id_col, target_col]).columns:
            fscores[column] = 100
        dump(fscores, open('output/fscores.pkl', 'wb'))
    else:
        fscores = load(open('output/fscores.pkl', 'rb'))

    n_features = len(fscores)
    features, probabilities = [feature for feature in fscores],  np.array([fscores[feature] for feature in fscores])
    probabilities = probabilities/probabilities.sum()

    n_selected_features = int(round(FEATURES_PCT*n_features))
    n_score_selected_features = int(round(SCORE_FEATURES_PCT*n_selected_features))
    n_uniform_selected_features = n_selected_features - n_score_selected_features

    score_selected_features = list(np.random.choice(features, size=n_score_selected_features, replace=False, p=probabilities))
    not_score_selected_features = [feature for feature in features if feature not in score_selected_features]

    uniform_selected_features = list(np.random.choice(not_score_selected_features, size=n_uniform_selected_features, replace=False))

    selected_features = score_selected_features + uniform_selected_features

    train = train_pkl[selected_features+[target_col]]

    X = train.drop(columns=[target_col])
    y = train[target_col]

    unbalance_factor = train[train[target_col]==0].shape[0]/train[train[target_col]==1].shape[0]
    del train
    collect()

    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    xgb_params = {
        'eta': 0.2,
        'colsample_bytree': 0.7,
        'colsample_bylevel':0.7,
        'min_child_weight':0,
        'alpha': 2,
        'max_depth': int(round(np.log(n_selected_features))),
        'scale_pos_weight': unbalance_factor/2,
        'eval_metric': 'auc',
        'random_state': int(time()),
        'silent': True,
        'grow_policy': 'lossguide',
        'tree_method': 'exact',
        'predictor': 'cpu_predictor',
        'objective': 'binary:logistic'
    }

    if use_gpu:
        xgb_params.update({'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'objective':'gpu:binary:logistic'})

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1368, random_state=int(time()), stratify=y)
    del X, y
    collect()

    dtrain = xgb.DMatrix(X_train, y_train)
    del X_train, y_train
    collect()

    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dvalid, 'validation')]

    model = xgb.train(xgb_params, dtrain, 500, watchlist, early_stopping_rounds = 15, verbose_eval=5)
    fscores_model = model.get_fscore()
    score = roc_auc_score(y_valid, model.predict(xgb.DMatrix(X_valid)))

    ##
    fscores_weight = model.get_score(importance_type='weight')
    weight_scores = [fscores_weight[feature] for feature in fscores_weight]
    print(min(weight_scores), max(weight_scores))

    fscores_gain = model.get_score(importance_type='gain')
    gain_scores = [fscores_gain[feature] for feature in fscores_gain]
    print(min(gain_scores), max(gain_scores))

    fscores_cover = model.get_score(importance_type='cover')
    cover_scores = [fscores_cover[feature] for feature in fscores_cover]
    print(min(cover_scores), max(cover_scores))
    ##

    del X_valid, y_valid
    collect()

    fscores = load(open('output/fscores.pkl', 'rb'))
    for feature in fscores_model:
        fscores[feature] += np.log(2+fscores_model[feature])
    dump(fscores, open('output/fscores.pkl', 'wb'))

    for feature in fscores:
        fscores[feature] = [fscores[feature]]
    fscores_df = pd.DataFrame(fscores).transpose().reset_index().rename(columns={'index':'feature', 0:'score'})
    fscores_df = fscores_df.sort_values('score', ascending=False)
    fscores_df.to_csv('output/fscores.csv', index=False)

    del dtrain, dvalid, fscores_df, fscores_model
    collect()

    test = load(open('intermediary/test.pkl', 'rb'))[ [id_col]+selected_features ]
    test[target_col] = model.predict(xgb.DMatrix(test.drop(columns=[id_col])))

    test[[id_col, target_col]].to_csv('output/pre_sub/{}_{}.csv'.format( int(time()), str(score).split('.')[1] ), index=False)
    del test, model
    collect()
