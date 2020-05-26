# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 23:51:56 2020

@author: mache
"""


import numpy as np
import pandas as pd
import os, warnings, random
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


train_df = pd.read_pickle('data-minification/train_df.pkl')
test_df = pd.read_pickle('data-minification/test_df.pkl')

remove_features = pd.read_pickle('data-minification/remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)

train_df.index = train_df['id'].tolist()
test_df.index = test_df['id'].tolist()
########################################################
# 交易金额组合特征
def uid_aggregation_dis(train_df, test_df, main_columns, uids):
    for main_column in main_columns:  
        for col in uids:
            print(col)
            new_col_name = col+'_'+main_column+'_'+'dis'
            temp_df = pd.concat([train_df[['id', col, main_column]], test_df[['id', col, main_column]]])
            for name, group in temp_df.groupby([col]):
                if len(group) > 10:
                    val_mean = group[main_column].mean()
                    group['dis'] = group[main_column].apply(lambda x: x - val_mean)
                    for i in range(len(group)):
                        ID = group['id'][i]
                        if ID in train_df.index.tolist():
                            train_df.loc[ID,  new_col_name] = group['dis'][i]
                        else:
                            test_df.loc[ID,  new_col_name] = group['dis'][i]
    return train_df, test_df

# --------------------------------------------------------
i_cols = ['ovd_30d_loan_tot_cnt', 'his_lng_ovd_day'
          ]
uids = ['loan_act_ind', 'pl_crd_lmt_cd'
        ]

train_df, test_df = uid_aggregation_dis(train_df, test_df, i_cols, uids)

########################################################
# 交易金额组合特征
def uid_aggregation(train_df, test_df, main_columns, uids, aggregations):
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col+'_'+main_column+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df)
                test_df[new_col_name]  = test_df[col].map(temp_df)
    return train_df, test_df

# --------------------------------------------------------
# i_cols = ['sum_trx_amt', 'real_sum_trx_amt', 
#           'l12mon_buy_fin_mng_whl_tms',
#           'trx_counts'
#           ]
i_cols = ['real_sum_trx_amt', 
          'l12mon_buy_fin_mng_whl_tms',
          'trx_counts'
          ]
uids = ['l1y_crd_card_csm_amt_dlm_cd', 'perm_crd_lmt_cd'
        ]
aggregations = ['mean']

train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

########################################################
# 理财产品购买次数分组后进行mean encode
# mean encoding
def mean_encoding(train_df, test_df, uids, self_encoding=False):
    for col in uids:
        new_col_name = col+'_'+'flag'+'_'+'mean'
        temp_df = train_df[[col, 'flag']]
        temp_df = temp_df.groupby([col])['flag'].agg(['mean']).reset_index().rename(
                                                        columns={'mean': new_col_name})

        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()   
        
        if self_encoding:
            train_df[col] = train_df[col].map(temp_df)
            test_df[col]  = test_df[col].map(temp_df)
        else:
            train_df[new_col_name] = train_df[col].map(temp_df).astype(float)
            test_df[new_col_name]  = test_df[col].map(temp_df).astype(float)
            
    return train_df, test_df

train_df, test_df = mean_encoding(train_df, test_df, ['job_year'], self_encoding=False)
##################################################################


##################################################################
# --------------------------------------------------------
c = 'l12mon_buy_fin_mng_whl_tms'
def group_fun(x):
    if x == np.nan:
        return x
    if x > 0:
        return 1
    else:
        return 0
train_df[c] = train_df[c].map(group_fun)
test_df[c] = test_df[c].map(group_fun)

c = 'l12_mon_fnd_buy_whl_tms'
def group_fun(x):
    if x == np.nan:
        return x
    if 21 <= x <= 25:
        return 21
    elif 26 <= x <= 30:
        return 26
    elif 31 <= x <= 35:
        return 31
    elif 36 <= x <= 40:
        return 36
    elif 41 <= x <= 50:
        return 41
    elif 51 <= x <= 80:
        return 51
    elif x > 80:
        return 81
    else:
        return x
train_df[c] = train_df[c].map(group_fun)
test_df[c] = test_df[c].map(group_fun)

c = 'l12_mon_insu_buy_whl_tms'
def group_fun(x):
    if x == np.nan:
        return x
    if x > 0:
        return 1
    else:
        return 0
train_df[c] = train_df[c].map(group_fun)
test_df[c] = test_df[c].map(group_fun)

c = 'l12_mon_gld_buy_whl_tms'
def group_fun(x):
    if x == np.nan:
        return x
    if x > 0:
        return 1
    else:
        return 0
train_df[c] = train_df[c].map(group_fun)
test_df[c] = test_df[c].map(group_fun)

c = 'age'
def group_fun(x):
    if x == np.nan:
        return x
    if x <= 65:
        return x
    elif 65 < x <= 70:
        return 68
    else:
        return 75
train_df['age_group'] = train_df[c].map(group_fun)
test_df['age_group'] = test_df[c].map(group_fun)
remove_features.append('age_group')
i_cols = ['pot_ast_lvl_cd'
          ]
uids = ['age_group'
        ]
# train_df, test_df = uid_aggregation_dis(train_df, test_df, i_cols, uids)
# train_df, test_df = uid_aggregation(train_df, test_df, i_cols, uids, aggregations)

c = 'ovd_30d_loan_tot_cnt'
def group_fun(x):
    if x == np.nan:
        return x
    if x > 0:
        return 1
    else:
        return 0
train_df[c] = train_df[c].map(group_fun)
test_df[c] = test_df[c].map(group_fun)

########################################################
# remove_features.extend(['l12mon_buy_fin_mng_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms'])
# remove_features.extend(['ic_ind', 'deg_cd', 'ovd_30d_loan_tot_cnt', 'l6mon_agn_ind', 'hav_hou_grp_ind'])

########################################################
def frequency_encoding(train_df, test_df, columns, self_encoding=False):
    for col in columns:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        if self_encoding:
            train_df[col] = train_df[col].map(fq_encode)
            test_df[col]  = test_df[col].map(fq_encode)            
        else:
            train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
            test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)
    return train_df, test_df

train_df, test_df = frequency_encoding(train_df, test_df, ['max_counts_page'], self_encoding=True)

########################################################
train_trd = pd.read_csv('训练数据集/训练数据集_trd.csv')
test_trd = pd.read_csv('评分数据集/评分数据集_trd.csv')
namelist = sorted(list(set(train_trd['Trx_Cod2_Cd'].drop_duplicates().tolist()).intersection(set(test_trd['Trx_Cod2_Cd'].drop_duplicates().tolist()))))

for name in namelist:
    remove_features.remove('Trx_Cod2_Cd_counts_'+str(name))

remove_features.extend(['d3', 'd4'])

for name in namelist:
    remove_features.remove('Trx_Cod2_Cd_sum_real_amt_'+str(name))
  
remove_features.extend(['d5', 'd6'])


########################################################

# remove_features.append('trx_sum_night')
# remove_features.extend(['page_tm_hour_dis'])

########################################################
train_trd = pd.read_csv('训练数据集/训练数据集_trd.csv')
test_trd = pd.read_csv('评分数据集/评分数据集_trd.csv')

def add_trd_feature(df, df_trd, df1, df2):
    df.index = df['id'].tolist()
    df_trd['trx_tm'] = pd.to_datetime(df_trd['trx_tm'])
    dic = dict()
    for name, group in df_trd.groupby('id'):
        dic[name] = group.reset_index()
    for i in range(len(df)):
        print(i/len(df))
        p = df['id'].tolist()[i]
        if p in dic:
            tempdf = dic[p]
            tempdf['trx_tm_m'] = tempdf['trx_tm'].map(lambda x: x.month)
            tempdf['trx_tm_d'] = tempdf['trx_tm'].map(lambda x: x.day)
            tempdf['trx_tm_h'] = tempdf['trx_tm'].map(lambda x: x.hour)
            df.loc[p, 'trx_days'] = len(tempdf['trx_tm_d'].value_counts()) / len(tempdf)
            df.loc[p, 'trx_days2'] = len(tempdf[tempdf['trx_tm_m']==5]['trx_tm_d'].value_counts()) / len(tempdf) + len(tempdf[tempdf['trx_tm_m']==6]['trx_tm_d'].value_counts()) / len(tempdf)   # 交易天数           
            
            tempdf['Dat_Flg1_Cd2'] = tempdf['Dat_Flg1_Cd'].map({'B':-1, 'C':1})
            tempdf['real_trx_amt'] = tempdf.apply(lambda x: x['Dat_Flg1_Cd2']*x['cny_trx_amt'], axis=1)
            dd_5 = tempdf[tempdf['trx_tm_m'] == 5]
            dd_6 = tempdf[tempdf['trx_tm_m'] == 6]
            df.loc[p, 'trx_realsum_5'] = dd_5['real_trx_amt'].sum()
            df.loc[p, 'trx_realsum_6'] = dd_6['real_trx_amt'].sum()
            df.loc[p, 'trx_counts_5'] = len(dd_5)
            df.loc[p, 'trx_counts_6'] = len(dd_6)
    return df
    
train_df = add_trd_feature(train_df, train_trd, train_trd, test_trd)
test_df = add_trd_feature(test_df, test_trd, train_trd, test_trd)

########################################################
# train_beh = pd.read_csv('训练数据集/训练数据集_beh.csv')
# test_beh = pd.read_csv('评分数据集/评分数据集_beh.csv')

# def add_beh_feature(df, df_beh, df1, df2):
#     df.index = df['id'].tolist()
#     df_beh['page_tm'] = pd.to_datetime(df_beh['page_tm'])
#     dic = dict()
#     for name, group in df_beh.groupby('id'):
#         dic[name] = group.reset_index()
#     for i in range(len(df)):
#         print(i/len(df))
#         p = df['id'].tolist()[i]
#         if p in dic:
#             tempdf = dic[p]
#             tempdf['page_tm_d'] = tempdf['page_tm'].map(lambda x: x.day)
#             df.loc[p, 'page_days'] = len(tempdf['page_tm_d'].value_counts()) / len(tempdf)   # 访问天数
#     return df

# train_df = add_beh_feature(train_df, train_beh, train_beh, test_beh)
# test_df = add_beh_feature(test_df, test_beh, train_beh, test_beh)

########################################################
def helper(df):
    df['cur_debit'] = df.apply(lambda x: x['cur_debit_cnt'] * x['cur_debit_min_opn_dt_cnt'], axis=1)
    df['cur_credit'] =  df.apply(lambda x: x['cur_credit_cnt'] * x['cur_credit_min_opn_dt_cnt'], axis=1)
    return df
train_df = helper(train_df)
test_df = helper(test_df)

########################################################
train_df.to_pickle('eda/train_df.pkl')
test_df.to_pickle('eda/test_df.pkl')
remove_features = pd.DataFrame(remove_features, columns=['features_to_remove'])
remove_features.to_pickle('eda/remove_features.pkl')
