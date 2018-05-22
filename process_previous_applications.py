from pickle import dump
import pandas as pd
from funcs import *

#aggrs = ['count', 'sum', 'max', 'min', 'std', 'skew', 'kurt']
aggrs = ['count', 'sum', 'max', 'min', 'std']

# PREPARING INPUT

previous_application = pd.read_csv('input/previous_application.csv').drop(columns=['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'SK_ID_PREV'])
credit_card_balance = pd.read_csv('input/credit_card_balance.csv').drop(columns=['SK_ID_PREV'])
installments_payments = pd.read_csv('input/installments_payments.csv').drop(columns=['SK_ID_PREV'])
pos_cash_balance = pd.read_csv('input/POS_CASH_balance.csv').drop(columns=['SK_ID_PREV'])

# ONE-HOT-ENCODING

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

credit_card_balance = pd.get_dummies(credit_card_balance, columns=['NAME_CONTRACT_STATUS'])

pos_cash_balance = pd.get_dummies(pos_cash_balance, columns=['NAME_CONTRACT_STATUS'])

# AGGREGATING, ENGINEERING AND RENAMING

previous_application = group_and_aggregate(previous_application, by='SK_ID_CURR', aggrs=aggrs)
previous_application['PCT_ANNUITY_(MEAN)'] = previous_application['AMT_ANNUITY_(MEAN)']/previous_application['AMT_CREDIT_(MEAN)']
previous_application['PCT_APPROVED_APPLICATION_(MEAN)'] = previous_application['AMT_CREDIT_(MEAN)']/previous_application['AMT_APPLICATION_(MEAN)']
previous_application['PCT_DOWN_PAYMENT_(MEAN)'] = previous_application['AMT_DOWN_PAYMENT_(MEAN)']/previous_application['AMT_CREDIT_(MEAN)']
previous_application['PCT_GOODS_PRICE_APPLICATION_(MEAN)'] = previous_application['AMT_GOODS_PRICE_(MEAN)']/previous_application['AMT_APPLICATION_(MEAN)']
previous_application['PCT_GOODS_PRICE_CREDIT_(MEAN)'] = previous_application['AMT_GOODS_PRICE_(MEAN)']/previous_application['AMT_CREDIT_(MEAN)']
rename_columns(previous_application, suffix='_(PREV_APP)', untouched=['SK_ID_CURR'])

installments_payments['PAYMENT_DELAY'] = installments_payments['DAYS_ENTRY_PAYMENT'] - installments_payments['DAYS_INSTALMENT']
installments_payments = group_and_aggregate(installments_payments, by='SK_ID_CURR', aggrs=aggrs)
installments_payments['PCT_PAYMENT_(MEAN)'] = installments_payments['AMT_PAYMENT_(MEAN)']/installments_payments['AMT_INSTALMENT_(MEAN)']
rename_columns(installments_payments, suffix='_(INST_PAYM)', untouched=['SK_ID_CURR'])

credit_card_balance = group_and_aggregate(credit_card_balance, by='SK_ID_CURR', aggrs=aggrs)
rename_columns(credit_card_balance, suffix='_(CRED_CARD)', untouched=['SK_ID_CURR'])

pos_cash_balance = group_and_aggregate(pos_cash_balance, by='SK_ID_CURR', aggrs=aggrs)
pos_cash_balance['PCT_INSTALMENT_FUTURE_(MEAN)'] = pos_cash_balance['CNT_INSTALMENT_FUTURE_(MEAN)']/pos_cash_balance['CNT_INSTALMENT_(MEAN)']
rename_columns(pos_cash_balance, suffix='_(POS_CASH)', untouched=['SK_ID_CURR'])

# OPTIMIZING

previous_application = optimize(previous_application)
installments_payments = optimize(installments_payments)
credit_card_balance = optimize(credit_card_balance)
pos_cash_balance = optimize(pos_cash_balance)

# DUMPING

dump(previous_application, open('intermediary/previous_application.pkl', 'wb'))
dump(installments_payments, open('intermediary/installments_payments.pkl', 'wb'))
dump(credit_card_balance, open('intermediary/credit_card_balance.pkl', 'wb'))
dump(pos_cash_balance, open('intermediary/pos_cash_balance.pkl', 'wb'))
