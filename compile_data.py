from pickle import load, dump
from gc import collect
import pandas as pd
from funcs import *

# PRE-MERGE FEATURE ENGINEERING

def pre_merge_eng(df):
    df['PCT_ANNUITY_CREDIT(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['PCT_ANNUITY_INCOME(ENG)'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
    df['PCT_CREDIT_INCOME(ENG)'] = df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
    df['PCT_GOODS_PRICE_INCOME(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_INCOME_TOTAL']
    df['PCT_GOODS_PRICE_CREDIT(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    df['AMT_INCOME_TOTAL_PER_CHILD(ENG)'] = df['AMT_INCOME_TOTAL']/df['CNT_CHILDREN']
    df['PCT_INCOME_TOTAL_PER_CHILD(ENG)'] = df['AMT_INCOME_TOTAL_PER_CHILD(ENG)']/df['AMT_INCOME_TOTAL']
    df['AMT_INCOME_TOTAL_PER_FAMILY_MEMBER(ENG)'] = df['AMT_INCOME_TOTAL']/df['CNT_FAM_MEMBERS']
    df['PCT_INCOME_TOTAL_PER_FAMILY_MEMBER(ENG)'] = df['AMT_INCOME_TOTAL_PER_FAMILY_MEMBER(ENG)']/df['AMT_INCOME_TOTAL']

# POS-MERGE FEATURE ENGINEERING

def pos_merge_eng(df):
    df['RATE_ANNUITY(ENG)(CURR_APP)(BUREAU)'] = df['AMT_ANNUITY(CURR_APP)']/df['AMT_ANNUITY(BUREAU)']
    df['RATE_CREDIT(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT(CURR_APP)']/df['AMT_CREDIT_SUM(BUREAU)']
    df['PCT_CREDIT_SUM(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_SUM_DEBT(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM_DEBT(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_SUM_OVERDUE(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM_OVERDUE(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_MAX_OVERDUE(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_MAX_OVERDUE(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_SUM_PER_REMAINING_DAY(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM_PER_REMAINING_DAY(ENG)(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)(CURR_APP)(BUREAU)'] = df['AMT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']
    df['PCT_ANNUITY(ENG)(CURR_APP)(BUREAU)'] = df['AMT_ANNUITY(BUREAU)']/df['AMT_INCOME_TOTAL(CURR_APP)']

    df['PCT_CREDIT_PER_MONTH(ENG)(CURR_APP)(PREV_APP)'] = df['AMT_CREDIT_PER_MONTH(ENG)(PREV_APP)']/df['AMT_INCOME_TOTAL(CURR_APP)']

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

# PRE-MERGE FEATURE ENGINEERING AND RENAMING

pre_merge_eng(application)
rename_columns(application, suffix='(CURR_APP)', untouched=['SK_ID_CURR', 'TARGET'])

# MERGING

for pkl_file in ['bureau', 'previous_application', 'installments_payments', 'credit_card_balance', 'pos_cash_balance']:
    application = pd.merge(application, load(open('intermediary/{}.pkl'.format(pkl_file), 'rb')), on='SK_ID_CURR', how='left')

# POS-MERGE FEATURE ENGINEERING

pos_merge_eng(application)

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
