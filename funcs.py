from scipy.stats import kurtosis
from gc import collect
import numpy as np

def optimize(df):
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

def rename_columns(df, suffix, untouched=[]):
    for column in df.drop(columns=untouched).columns:
        df.rename(columns={column:column+suffix}, inplace=True)

def group_and_aggregate(df, by, aggrs):
    df_groupby = df.groupby(by)
    df_by = df_groupby.mean().reset_index()
    for column in df.drop(columns=[by]).columns:
        for aggr in aggrs:
            if aggr!='kurt':
                df_by[column+'_({})'.format(aggr.upper())] = df_groupby[column].transform(aggr)
            else:
                df_by[column+'_({})'.format(aggr.upper())] = df_groupby[column].apply(kurtosis)
    return df_by
