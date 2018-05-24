from pickle import dump
from gc import collect
import pandas as pd
from funcs import *

#aggrs = ['count', 'sum', 'max', 'min', 'std', 'var', 'skew', 'kurt']
#aggrs = ['count', 'sum', 'max', 'min', 'std', 'var']
aggrs = []

##### feature engineering

def eng_bureau(df):
    df['PCT_CREDIT_SUM_DEBT(ENG)'] = df['AMT_CREDIT_SUM_DEBT']/df['AMT_CREDIT_SUM']
    df['PCT_CREDIT_SUM_DEBT_LIMIT(ENG)'] = df['AMT_CREDIT_SUM_DEBT']/df['AMT_CREDIT_SUM_LIMIT']
    df['PCT_CREDIT_SUM_OVERDUE(ENG)'] = df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM']
    df['PCT_CREDIT_SUM_OVERDUE_LIMIT(ENG)'] = df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM_LIMIT']
    df['PCT_CREDIT_MAX_OVERDUE(ENG)'] = df['AMT_CREDIT_MAX_OVERDUE']/df['AMT_CREDIT_SUM']
    df['PCT_CREDIT_MAX_OVERDUE_LIMIT(ENG)'] = df['AMT_CREDIT_MAX_OVERDUE']/df['AMT_CREDIT_SUM_LIMIT']
    df['PCT_ANNUITY(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT_SUM']
    df['AMT_CREDIT_SUM_PER_REMAINING_DAY(ENG)'] = df['AMT_CREDIT_SUM']/df['DAYS_CREDIT_ENDDATE']
    df['AMT_CREDIT_SUM_DEBT_PER_REMAINING_DAY(ENG)'] = df['AMT_CREDIT_SUM_DEBT']/df['DAYS_CREDIT_ENDDATE']
    df['AMT_CREDIT_SUM_OVERDUE_PER_REMAINING_DAY(ENG)'] = df['AMT_CREDIT_SUM_OVERDUE']/df['DAYS_CREDIT_ENDDATE']

def eng_previous_application(df):
    df['PCT_ANNUITY(ENG)'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['PCT_APPROVED_APPLICATION(ENG)'] = df['AMT_CREDIT']/df['AMT_APPLICATION']
    df['PCT_DOWN_PAYMENT(ENG)'] = df['AMT_DOWN_PAYMENT']/df['AMT_CREDIT']
    df['PCT_GOODS_PRICE_APPLICATION(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_APPLICATION']
    df['PCT_GOODS_PRICE_CREDIT(ENG)'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    df['AMT_CREDIT_PER_MONTH(ENG)'] = df['AMT_CREDIT']/df['CNT_PAYMENT']

def eng_pos_cash_balance(df):
    df['PCT_INSTALMENT_FUTURE(ENG)'] = df['CNT_INSTALMENT_FUTURE']/df['CNT_INSTALMENT']
    df['PCT_SK_DPD_DEF(ENG)'] = df['SK_DPD_DEF']/df['SK_DPD']

def eng_installments_payments(df):
    df['PAYMENT_DELAY(ENG)'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['PCT_PAYMENT(ENG)'] = df['AMT_PAYMENT']/df['AMT_INSTALMENT']

def eng_credit_card_balance(df):
    for feature in ['DRAWINGS_ATM_CURRENT', 'DRAWINGS_CURRENT', 'DRAWINGS_OTHER_CURRENT', 'DRAWINGS_POS_CURRENT']:
        df['PCT_'+feature+'(ENG)'] = df['AMT_'+feature]/df['AMT_CREDIT_LIMIT_ACTUAL']
        df['PCT_'+feature+'_PER_DRAWING(ENG)'] = df['PCT_'+feature+'(ENG)']/df['CNT_'+feature]
    df['PCT_BALANCE(ENG)'] = df['AMT_BALANCE']/df['AMT_CREDIT_LIMIT_ACTUAL']
    df['TAX_RATE(ENG)'] = df['AMT_BALANCE']/df['AMT_RECEIVABLE_PRINCIPAL']
    df['PCT_PAYMENT_CURRENT(ENG)'] = df['AMT_PAYMENT_CURRENT']/df['AMT_BALANCE']
    df['PCT_PAYMENT_TOTAL_CURRENT(ENG)'] = df['AMT_PAYMENT_TOTAL_CURRENT']/df['AMT_BALANCE']

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
rename_columns(bureau_balance, suffix='(BUR_BAL)', untouched=['SK_ID_CURR'])
eng_bureau(bureau)
bureau = group_and_aggregate(bureau, by='SK_ID_CURR', aggrs=aggrs)
eng_bureau(bureau)
rename_columns(bureau, suffix='(BUREAU)', untouched=['SK_ID_CURR'])
bureau = bureau.merge(bureau_balance, on='SK_ID_CURR', how='left')
dump(bureau, open('intermediary/bureau.pkl', 'wb'))
del bureau, bureau_balance
collect()

##### credit_card_balance

credit_card_balance = pd.read_csv('input/credit_card_balance.csv').drop(columns=['SK_ID_PREV', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE'])
credit_card_balance = pd.get_dummies(credit_card_balance, columns=['NAME_CONTRACT_STATUS'])
eng_credit_card_balance(credit_card_balance)
credit_card_balance = group_and_aggregate(credit_card_balance, by='SK_ID_CURR', aggrs=aggrs)
eng_credit_card_balance(credit_card_balance)
rename_columns(credit_card_balance, suffix='(CRED_CARD)', untouched=['SK_ID_CURR'])
dump(credit_card_balance, open('intermediary/credit_card_balance.pkl', 'wb'))
del credit_card_balance
collect()

##### installments_payments

installments_payments = pd.read_csv('input/installments_payments.csv').drop(columns=['SK_ID_PREV'])
eng_installments_payments(installments_payments)
installments_payments = group_and_aggregate(installments_payments, by='SK_ID_CURR', aggrs=aggrs)
eng_installments_payments(installments_payments)
rename_columns(installments_payments, suffix='(INST_PAYM)', untouched=['SK_ID_CURR'])
dump(installments_payments, open('intermediary/installments_payments.pkl', 'wb'))
del installments_payments
collect()

##### pos_cash_balance

pos_cash_balance = pd.read_csv('input/POS_CASH_balance.csv').drop(columns=['SK_ID_PREV'])
pos_cash_balance = pd.get_dummies(pos_cash_balance, columns=['NAME_CONTRACT_STATUS'])
eng_pos_cash_balance(pos_cash_balance)
pos_cash_balance = group_and_aggregate(pos_cash_balance, by='SK_ID_CURR', aggrs=aggrs)
eng_pos_cash_balance(pos_cash_balance)
rename_columns(pos_cash_balance, suffix='(POS_CASH)', untouched=['SK_ID_CURR'])
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
eng_previous_application(previous_application)
previous_application = group_and_aggregate(previous_application, by='SK_ID_CURR', aggrs=aggrs)
eng_previous_application(previous_application)
rename_columns(previous_application, suffix='(PREV_APP)', untouched=['SK_ID_CURR'])
dump(previous_application, open('intermediary/previous_application.pkl', 'wb'))
