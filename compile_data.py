from pickle import load, dump
from gc import collect
import pandas as pd
from funcs import *

# PREPARING INPUT

application_train = pd.read_csv('input/application_train.csv')
application_test = pd.read_csv('input/application_test.csv')
application_test['TARGET'] = 2

columns = list(application_train.columns)
columns.remove('SK_ID_CURR')
columns.remove('TARGET')
columns = ['SK_ID_CURR'] + columns + ['TARGET']

application_train = application_train[columns]
application_test = application_test[columns]

application = pd.concat([application_train, application_test], ignore_index=True)
del application_train, application_test
collect()

# ONE-HOT-ENCODING

application = pd.get_dummies(application,
                             drop_first=True,
                             dummy_na=True,
                             columns=[
                                 'FONDKAPREMONT_MODE',
                                 'WALLSMATERIAL_MODE',
                                 'HOUSETYPE_MODE',
                                 'EMERGENCYSTATE_MODE',
                                 'OCCUPATION_TYPE',
                                 'NAME_TYPE_SUITE'
                             ])

application = pd.get_dummies(application,
                             drop_first=True,
                             columns=[
                                 'ORGANIZATION_TYPE',
                                 'WEEKDAY_APPR_PROCESS_START',
                                 'CODE_GENDER',
                                 'FLAG_OWN_REALTY',
                                 'FLAG_OWN_CAR',
                                 'NAME_CONTRACT_TYPE',
                                 'NAME_EDUCATION_TYPE',
                                 'NAME_FAMILY_STATUS',
                                 'NAME_HOUSING_TYPE',
                                 'NAME_INCOME_TYPE'
                             ])

# ENGINEERING AND RENAMING

application['PCT_ANNUITY_CREDIT'] = application['AMT_ANNUITY']/application['AMT_CREDIT']
application['PCT_ANNUITY_INCOME'] = application['AMT_ANNUITY']/application['AMT_INCOME_TOTAL']
application['PCT_CREDIT_INCOME'] = application['AMT_CREDIT']/application['AMT_INCOME_TOTAL']
application['PCT_GOODS_PRICE_INCOME'] = application['AMT_GOODS_PRICE']/application['AMT_INCOME_TOTAL']
application['PCT_GOODS_PRICE_CREDIT'] = application['AMT_GOODS_PRICE']/application['AMT_CREDIT']
rename_columns(application, suffix='_(APP)', untouched=['SK_ID_CURR', 'TARGET'])

# MERGING

for pkl_file in ['bureau', 'previous_application', 'installments_payments', 'credit_card_balance', 'pos_cash_balance']:
    application = pd.merge(application, load(open('intermediary/{}.pkl'.format(pkl_file), 'rb')), on='SK_ID_CURR', how='left')

# FEATURE ENGINEERING

# FILLING MISSING VALUES

for column in application.columns:
    minn = application[column].min()
    maxx = application[column].max()
    application[column].fillna(min(-10, minn)-max(10, 10*(maxx-minn)), inplace=True)

# OPTIMIZING

application = optimize(application)

# DUMPING

dump(application[application['TARGET']!=2].reset_index(drop=True), open('intermediary/train.pkl', 'wb'))
dump(application[application['TARGET']==2].drop(columns=['TARGET']).reset_index(drop=True), open('intermediary/test.pkl', 'wb'))
