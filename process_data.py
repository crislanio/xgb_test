from pickle import dump
from gc import collect
import pandas as pd
from funcs import *

aggrs = ['count', 'sum', 'max', 'min', 'std', 'var', 'skew', 'kurt']
#aggrs = ['count', 'sum', 'max', 'min', 'std', 'var']
#aggrs = []

##### bureau

bureau = pd.read_csv('input/bureau.csv')
bureau_balance = pd.read_csv('input/bureau_balance.csv')

bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates(subset=['SK_ID_BUREAU']),
                                        on='SK_ID_BUREAU',
                                        how='inner').drop(columns=['SK_ID_BUREAU'])
del bureau['SK_ID_BUREAU']

bureau = pd.get_dummies(bureau, columns=['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'])
bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS'])
bureau_balance = group_and_aggregate(bureau_balance, by='SK_ID_CURR', aggrs=aggrs)
rename_columns(bureau_balance, suffix='_(BUR_BAL)', untouched=['SK_ID_CURR'])
bureau = group_and_aggregate(bureau, by='SK_ID_CURR', aggrs=aggrs)
bureau['PCT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT']/bureau['AMT_CREDIT_SUM']
bureau['PCT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE']/bureau['AMT_CREDIT_SUM']
bureau['PCT_CREDIT_MAX_OVERDUE'] = bureau['AMT_CREDIT_MAX_OVERDUE']/bureau['AMT_CREDIT_SUM']
bureau['PCT_ANNUITY'] = bureau['AMT_ANNUITY']/bureau['AMT_CREDIT_SUM']
rename_columns(bureau, suffix='_(BUREAU)', untouched=['SK_ID_CURR'])
bureau = bureau.merge(bureau_balance, on='SK_ID_CURR', how='left')
dump(bureau, open('intermediary/bureau.pkl', 'wb'))
del bureau, bureau_balance
collect()

##### credit_card_balance

credit_card_balance = pd.read_csv('input/credit_card_balance.csv').drop(columns=['SK_ID_PREV', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE'])
credit_card_balance = pd.get_dummies(credit_card_balance, columns=['NAME_CONTRACT_STATUS'])
credit_card_balance = group_and_aggregate(credit_card_balance, by='SK_ID_CURR', aggrs=aggrs)
for feature in ['DRAWINGS_ATM_CURRENT', 'DRAWINGS_CURRENT', 'DRAWINGS_OTHER_CURRENT', 'DRAWINGS_POS_CURRENT']:
    credit_card_balance['PCT_'+feature] = credit_card_balance['AMT_'+feature]/credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
    credit_card_balance['PCT_'+feature+'_PER_DRAWING'] = credit_card_balance['PCT_'+feature]/credit_card_balance['CNT_'+feature]
credit_card_balance['PCT_BALANCE'] = credit_card_balance['AMT_BALANCE']/credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL']
credit_card_balance['TAX_RATE'] = credit_card_balance['AMT_BALANCE']/credit_card_balance['AMT_RECEIVABLE_PRINCIPAL']
credit_card_balance['PCT_PAYMENT_CURRENT'] = credit_card_balance['AMT_PAYMENT_CURRENT']/credit_card_balance['AMT_BALANCE']
credit_card_balance['PCT_PAYMENT_TOTAL_CURRENT'] = credit_card_balance['AMT_PAYMENT_TOTAL_CURRENT']/credit_card_balance['AMT_BALANCE']
rename_columns(credit_card_balance, suffix='_(CRED_CARD)', untouched=['SK_ID_CURR'])
dump(credit_card_balance, open('intermediary/credit_card_balance.pkl', 'wb'))
del credit_card_balance
collect()

##### installments_payments

installments_payments = pd.read_csv('input/installments_payments.csv').drop(columns=['SK_ID_PREV'])
installments_payments['PAYMENT_DELAY'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
installments_payments = group_and_aggregate(installments_payments, by='SK_ID_CURR', aggrs=aggrs)
installments_payments['PCT_PAYMENT'] = installments_payments['AMT_PAYMENT']/installments_payments['AMT_INSTALMENT']
rename_columns(installments_payments, suffix='_(INST_PAYM)', untouched=['SK_ID_CURR'])
dump(installments_payments, open('intermediary/installments_payments.pkl', 'wb'))
del installments_payments
collect()

##### pos_cash_balance

pos_cash_balance = pd.read_csv('input/POS_CASH_balance.csv').drop(columns=['SK_ID_PREV'])
pos_cash_balance = pd.get_dummies(pos_cash_balance, columns=['NAME_CONTRACT_STATUS'])
pos_cash_balance = group_and_aggregate(pos_cash_balance, by='SK_ID_CURR', aggrs=aggrs)
pos_cash_balance['PCT_INSTALMENT_FUTURE'] = pos_cash_balance['CNT_INSTALMENT_FUTURE']/pos_cash_balance['CNT_INSTALMENT']
rename_columns(pos_cash_balance, suffix='_(POS_CASH)', untouched=['SK_ID_CURR'])
dump(pos_cash_balance, open('intermediary/pos_cash_balance.pkl', 'wb'))
del pos_cash_balance
collect()

##### previous_application

previous_application = pd.read_csv('input/previous_application.csv').drop(columns=['SK_ID_PREV'])
previous_application = pd.get_dummies(previous_application,
                                      columns=[
                                          'NAME_PRODUCT_TYPE',
                                          'NAME_YIELD_GROUP',
                                          'NAME_SELLER_INDUSTRY',
                                          'NAME_CONTRACT_STATUS',
                                          'NAME_PORTFOLIO',
                                          'NAME_PAYMENT_TYPE',
                                          'NAME_GOODS_CATEGORY',
                                          'NAME_CONTRACT_TYPE',
                                          'NAME_CLIENT_TYPE',
                                          'NAME_CASH_LOAN_PURPOSE',
                                          'FLAG_LAST_APPL_PER_CONTRACT',
                                          'CODE_REJECT_REASON',
                                          'CHANNEL_TYPE',
                                          'WEEKDAY_APPR_PROCESS_START'
                                      ])
previous_application = pd.get_dummies(previous_application,
                                      dummy_na=True,
                                      columns=[
                                          'NAME_TYPE_SUITE',
                                          'PRODUCT_COMBINATION'
                                      ])
previous_application = group_and_aggregate(previous_application, by='SK_ID_CURR', aggrs=aggrs)
previous_application['PCT_ANNUITY'] = previous_application['AMT_ANNUITY']/previous_application['AMT_CREDIT']
previous_application['PCT_APPROVED_APPLICATION'] = previous_application['AMT_CREDIT']/previous_application['AMT_APPLICATION']
previous_application['PCT_DOWN_PAYMENT'] = previous_application['AMT_DOWN_PAYMENT']/previous_application['AMT_CREDIT']
previous_application['PCT_GOODS_PRICE_APPLICATION'] = previous_application['AMT_GOODS_PRICE']/previous_application['AMT_APPLICATION']
previous_application['PCT_GOODS_PRICE_CREDIT'] = previous_application['AMT_GOODS_PRICE']/previous_application['AMT_CREDIT']
rename_columns(previous_application, suffix='_(PREV_APP)', untouched=['SK_ID_CURR'])
dump(previous_application, open('intermediary/previous_application.pkl', 'wb'))
