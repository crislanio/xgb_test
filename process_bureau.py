from pickle import dump
import pandas as pd
from funcs import *

#aggrs = ['count', 'sum', 'max', 'min', 'std', 'skew', 'kurt']
aggrs = ['count', 'sum', 'max', 'min', 'std']

# PREPARING INPUT

bureau = pd.read_csv('input/bureau.csv')
bureau_balance = pd.read_csv('input/bureau_balance.csv')

bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates(subset=['SK_ID_BUREAU']),
                                        on='SK_ID_BUREAU',
                                        how='inner').drop(columns=['SK_ID_BUREAU'])
del bureau['SK_ID_BUREAU']

# ONE-HOT-ENCODING

bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'])
bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS'])

# AGGREGATING, ENGINEERING AND RENAMING

bureau_balance = group_and_aggregate(bureau_balance, by='SK_ID_CURR', aggrs=aggrs)
rename_columns(bureau_balance, suffix='_(BUR_BAL)', untouched=['SK_ID_CURR'])

bureau = group_and_aggregate(bureau, by='SK_ID_CURR', aggrs=aggrs)
bureau['PCT_CREDIT_SUM_DEBT_(MEAN)'] = bureau['AMT_CREDIT_SUM_DEBT_(MEAN)']/bureau['AMT_CREDIT_SUM_(MEAN)']
bureau['PCT_CREDIT_SUM_OVERDUE_(MEAN)'] = bureau['AMT_CREDIT_SUM_OVERDUE_(MEAN)']/bureau['AMT_CREDIT_SUM_(MEAN)']
bureau['PCT_CREDIT_MAX_OVERDUE_(MEAN)'] = bureau['AMT_CREDIT_MAX_OVERDUE_(MEAN)']/bureau['AMT_CREDIT_SUM_(MEAN)']
bureau['PCT_ANNUITY_(MEAN)'] = bureau['AMT_ANNUITY_(MEAN)']/bureau['AMT_CREDIT_SUM_(MEAN)']
rename_columns(bureau, suffix='_(BUREAU)', untouched=['SK_ID_CURR'])

# MERGING AND OPTIMIZING

bureau = optimize(bureau.merge(bureau_balance, on='SK_ID_CURR', how='left'))

# DUMPING

dump(bureau, open('intermediary/bureau.pkl', 'wb'))
