# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:12:16 2020

@author: mache
"""


import numpy as np
import pandas as pd
import os, warnings, random
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


## -------------------

train_tag = pd.read_csv('训练数据集/训练数据集_tag.csv')
test_tag = pd.read_csv('评分数据集/评分数据集_tag.csv')
remove_features = []
remove_features.extend(['id', 'flag'])



print('train_tag shape is {}'.format(train_tag.shape))
print('test_tag shape is {}'.format(test_tag.shape))


def minify_identity_df(df):
    df['gdr_cd'] = df['gdr_cd'].map({'M':1, 'F':0})
    df['job_year'] = df['job_year'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['frs_agn_dt_cnt'] = df['frs_agn_dt_cnt'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['fin_rsk_ases_grd_cd'] = df['fin_rsk_ases_grd_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['confirm_rsk_ases_lvl_typ_cd'] = df['confirm_rsk_ases_lvl_typ_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['cust_inv_rsk_endu_lvl_cd'] = df['cust_inv_rsk_endu_lvl_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['tot_ast_lvl_cd'] = df['tot_ast_lvl_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['pot_ast_lvl_cd'] = df['pot_ast_lvl_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l12mon_buy_fin_mng_whl_tms'] = df['l12mon_buy_fin_mng_whl_tms'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l12_mon_fnd_buy_whl_tms'] = df['l12_mon_fnd_buy_whl_tms'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l12_mon_insu_buy_whl_tms'] = df['l12_mon_insu_buy_whl_tms'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l12_mon_gld_buy_whl_tms'] = df['l12_mon_gld_buy_whl_tms'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['pl_crd_lmt_cd'] = df['pl_crd_lmt_cd'].apply(int)
    df['ovd_30d_loan_tot_cnt'] = df['ovd_30d_loan_tot_cnt'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['his_lng_ovd_day'] = df['his_lng_ovd_day'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['hld_crd_card_grd_cd'] = df['hld_crd_card_grd_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l1y_crd_card_csm_amt_dlm_cd'] = df['l1y_crd_card_csm_amt_dlm_cd'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['ic_ind'] = df['ic_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['fr_or_sh_ind'] = df['fr_or_sh_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['dnl_mbl_bnk_ind'] = df['dnl_mbl_bnk_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['dnl_bind_cmb_lif_ind'] = df['dnl_bind_cmb_lif_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['hav_car_grp_ind'] = df['hav_car_grp_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['hav_hou_grp_ind'] = df['hav_hou_grp_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['l6mon_agn_ind'] = df['l6mon_agn_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['vld_rsk_ases_ind'] = df['vld_rsk_ases_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['loan_act_ind'] = df['loan_act_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['crd_card_act_ind'] = df['crd_card_act_ind'].apply(lambda x: np.nan if str(x) == '\\N' else int(x))
    df['atdd_type'] = df['atdd_type'].apply(lambda x: float(x) if (x != np.nan and str(x) != '\\N') else np.nan)
    df['mrg_situ_cd'] = df['mrg_situ_cd'].apply(lambda x: np.nan if str(x) == '\\N' else x)
    df['edu_deg_cd'] = df['edu_deg_cd'].apply(lambda x: np.nan if str(x) == '\\N' else x)
    df['acdm_deg_cd'] = df['acdm_deg_cd'].apply(lambda x: np.nan if str(x) == '\\N' else x)
    df['deg_cd'] = df['deg_cd'].apply(lambda x: np.nan if str(x) == '\\N' else x)
    return df

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
            train_df[col] = train_df[col].map(temp_df).astype(float)
            test_df[col]  = test_df[col].map(temp_df).astype(float)
        else:
            train_df[new_col_name] = train_df[col].map(temp_df).astype(float)
            test_df[new_col_name]  = test_df[col].map(temp_df).astype(float)
            
    return train_df, test_df

#================================tag====================================

train_tag = minify_identity_df(train_tag)
test_tag = minify_identity_df(test_tag)
    

for col in ['mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd']:
    if train_tag[col].dtype=='O':
        print(col)
        train_tag[col] = train_tag[col].fillna('unseen_before_label')
        test_tag[col]  = test_tag[col].fillna('unseen_before_label')
        
        train_tag[col] = train_tag[col].astype(str)
        test_tag[col] = test_tag[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_tag[col])+list(test_tag[col]))
        train_tag[col] = le.transform(train_tag[col])
        test_tag[col]  = le.transform(test_tag[col])
        
        train_tag[col] = train_tag[col].astype('category')
        test_tag[col] = test_tag[col].astype('category')

#================================trd====================================
train_df = train_tag
test_df = test_tag
del train_tag, test_tag

train_trd = pd.read_csv('训练数据集/训练数据集_trd.csv')
test_trd = pd.read_csv('评分数据集/评分数据集_trd.csv')

print('train_trd shape is {}'.format(train_trd.shape))
print('test_trd shape is {}'.format(test_trd.shape))

def add_trd_feature(df, df_trd, df1, df2):
    df.index = df['id'].tolist()
    df_trd['trx_tm'] = pd.to_datetime(df_trd['trx_tm'])
    df_trd['trx_tm_hour'] = (df_trd['trx_tm'].dt.hour).astype(np.int8)
    hot_hour = pd.concat([df1, df2])['trx_tm_hour'].value_counts().idxmax()
    dic = dict()
    for name, group in df_trd.groupby('id'):
        dic[name] = group.reset_index()
    for i in range(len(df)):
        print(i/len(df))
        p = df['id'].tolist()[i]
        if p in dic:
            tempdf = dic[p]
            tempdf['Dat_Flg1_Cd2'] = tempdf['Dat_Flg1_Cd'].map({'B':-1, 'C':1})
            tempdf['real_trx_amt'] = tempdf.apply(lambda x: x['Dat_Flg1_Cd2']*x['cny_trx_amt'], axis=1)
            df.loc[p, 'trx_tm_hour_dis'] = tempdf['trx_tm_hour'].apply(lambda x: min(x-hot_hour, 24-x+hot_hour)).mean() # 距最火交易小时的平均距离
            df.loc[p, 'trx_counts'] = len(tempdf) # 交易次数
            df.loc[p, 'trx_counts_night'] = len(tempdf[(tempdf['trx_tm_hour'] >= 22) | (tempdf['trx_tm_hour'] <= 4)])# 夜晚交易次数
            df.loc[p, 'trx_realsum_night'] = tempdf[(tempdf['trx_tm_hour'] >= 22) | (tempdf['trx_tm_hour'] <= 4)]['real_trx_amt'].sum() # 夜晚净交易额
            df.loc[p, 'trx_sum_night'] = tempdf[(tempdf['trx_tm_hour'] >= 22) | (tempdf['trx_tm_hour'] <= 4)]['cny_trx_amt'].sum() # 夜晚交易额
            
            tempdf['trx_tm_m'] = tempdf['trx_tm'].map(lambda x: x.month)
            tempdf['trx_tm_d'] = tempdf['trx_tm'].map(lambda x: x.day)
            dd = tempdf[(((tempdf['trx_tm_m'] == 5) & (tempdf['trx_tm_d'].isin([1, 2, 3, 4, 11, 12, 18, 19, 25, 26]))) | 
                         ((tempdf['trx_tm_m'] == 6) & (tempdf['trx_tm_d'].isin([1, 2, 7, 8, 9, 15, 16, 22, 23, 29, 30]))))]
            dd_51 = tempdf[((tempdf['trx_tm_m'] == 5) & (tempdf['trx_tm_d'].isin([1, 2, 3, 4])))]
            df.loc[p, 'trx_counts_holiday'] = len(dd) # 节假日交易次数
            df.loc[p, 'trx_realsum_holiday'] = dd['real_trx_amt'].sum()  # 节假日净交易额
            df.loc[p, 'trx_sum_holiday'] = dd['cny_trx_amt'].sum()  # 节假日交易额 
            df.loc[p, 'trx_counts_51'] = len(dd_51) # 51节假日交易次数
            df.loc[p, 'trx_realsum_51'] = dd_51['real_trx_amt'].sum()  # 51节假日净交易额
            df.loc[p, 'trx_sum_51'] = dd_51['cny_trx_amt'].sum()  # 51节假日交易额 
            
            df.loc[p, 'sum_trx_amt'] = tempdf['cny_trx_amt'].sum() # 总交易额
            df.loc[p, 'real_sum_trx_amt'] = tempdf['real_trx_amt'].sum() # 总净交易额
            df.loc[p, 'max_trx_amt'] = tempdf['cny_trx_amt'].max() # 最大正交易额
            df.loc[p, 'mmin_trx_amt'] = tempdf['cny_trx_amt'].min() # 最大负交易额
            for c in ['Dat_Flg1_Cd']:
                namelist = ['B', 'C']
                dic1 = dict()
                for name, group in tempdf.groupby(c):
                    dic1[name] = group.reset_index()
                for name in namelist:
                    if name in dic1:
                        length = len(dic1[name])                
                        c_sum = dic1[name]['cny_trx_amt'].sum() 
                    else:
                        length = 0
                        c_sum = 0
                    df.loc[p, c+'_counts_'+str(name)] = length    # 该分类的交易数量
                    df.loc[p, c+'_sum_amt_'+str(name)] = c_sum     # 该分类的总交易额
            for c in ['Dat_Flg3_Cd', 'Trx_Cod1_Cd']:
                namelist = sorted(list(set(df1[c].drop_duplicates().tolist()).intersection(set(df2[c].drop_duplicates().tolist()))))
                dic1 = dict()
                for name, group in tempdf.groupby(c):
                    dic1[name] = group.reset_index()
                for name in namelist:
                    if name in dic1:
                        length = len(dic1[name])   
                        c_sum = dic1[name]['real_trx_amt'].sum() 
                    else:
                        length = 0
                        c_sum =0
                    df.loc[p, c+'_counts_'+str(name)] = length    # 该分类的交易数量
                    df.loc[p, c+'_sum_real_amt_'+str(name)] = c_sum     # 该分类的总净交易额
            for c in ['Trx_Cod2_Cd']:
                namelist = sorted(list(set(df1[c].drop_duplicates().tolist()).intersection(set(df2[c].drop_duplicates().tolist()))))
                dic1 = dict()
                for name, group in tempdf.groupby(c):
                    dic1[name] = group.reset_index()
                for name in namelist:
                    if name in dic1:
                        length = len(dic1[name])   
                        c_sum = dic1[name]['real_trx_amt'].sum() 
                    else:
                        length = 0
                        c_sum =0
                    df.loc[p, c+'_counts_'+str(name)] = length    # 该分类的交易数量
                    df.loc[p, c+'_sum_real_amt_'+str(name)] = c_sum     # 该分类的总净交易额
    return df
    
train_df = add_trd_feature(train_df, train_trd, train_trd, test_trd)
test_df = add_trd_feature(test_df, test_trd, train_trd, test_trd)

# PCA降维
namelist = sorted(list(set(train_trd['Trx_Cod2_Cd'].drop_duplicates().tolist()).intersection(set(test_trd['Trx_Cod2_Cd'].drop_duplicates().tolist()))))

pca_list = []
for name in namelist:
    remove_features.append('Trx_Cod2_Cd_counts_'+str(name))
    pca_list.append('Trx_Cod2_Cd_counts_'+str(name))
pca=PCA(n_components=2)
train_df[pca_list] = train_df[pca_list].fillna(0)
test_df[pca_list] = test_df[pca_list].fillna(0)
pca.fit(train_df[pca_list])
temp_train = pd.DataFrame(pca.transform(train_df[pca_list]))
temp_train.index = train_df.index.tolist()
train_df[['d3', 'd4']] = temp_train
temp_test = pd.DataFrame(pca.transform(test_df[pca_list]))
temp_test.index = test_df.index.tolist()
test_df[['d3', 'd4']] = temp_test

pca_list = []
for name in namelist:
    remove_features.append('Trx_Cod2_Cd_sum_real_amt_'+str(name))
    pca_list.append('Trx_Cod2_Cd_sum_real_amt_'+str(name))
pca=PCA(n_components=2)
train_df[pca_list] = train_df[pca_list].fillna(0)
test_df[pca_list] = test_df[pca_list].fillna(0)
pca.fit(train_df[pca_list])
temp_train = pd.DataFrame(pca.transform(train_df[pca_list]))
temp_train.index = train_df.index.tolist()
train_df[['d5', 'd6']] = temp_train
temp_test = pd.DataFrame(pca.transform(test_df[pca_list]))
temp_test.index = test_df.index.tolist()
test_df[['d5', 'd6']] = temp_test

#================================beh====================================
train_beh = pd.read_csv('训练数据集/训练数据集_beh.csv')
test_beh = pd.read_csv('评分数据集/评分数据集_beh.csv')

def add_beh_feature(df, df_beh, df1, df2):
    df.index = df['id'].tolist()
    df_beh['page_tm'] = pd.to_datetime(df_beh['page_tm'])
    df_beh['page_tm_hour'] = (df_beh['page_tm'].dt.hour).astype(np.int8)
    hot_hour = pd.concat([df1, df2])['page_tm_hour'].value_counts().idxmax()
    dic = dict()
    for name, group in df_beh.groupby('id'):
        dic[name] = group.reset_index()
    for i in range(len(df)):
        print(i/len(df))
        p = df['id'].tolist()[i]
        if p in dic:
            tempdf = dic[p]
            df.loc[p, 'page_tm_hour_dis'] = tempdf['page_tm_hour'].apply(lambda x: min(x-hot_hour, 24-x+hot_hour)).mean() # 距最火访问小时的平均距离
            df.loc[p, 'beh_counts'] = len(tempdf)           # 访问次数           
            df.loc[p, 'max_counts_page'] = tempdf['page_no'].value_counts().idxmax()  # 最常访问页码 
            namelist = sorted(list(set(df1['page_no'].drop_duplicates().tolist()).intersection(set(df2['page_no'].drop_duplicates().tolist()))))
            dic1 = dict()
            for name, group in tempdf.groupby('page_no'):
                dic1[name] = group.reset_index()
            for name in namelist:
                if name in dic1:
                    length = len(dic1[name])
                else:
                    length = 0
                df.loc[p, 'page_no_counts_'+str(name)] = length            # 该页码的访问次数
    return df

train_df = add_beh_feature(train_df, train_beh, train_beh, test_beh)
test_df = add_beh_feature(test_df, test_beh, train_beh, test_beh)
remove_features.extend(['max_counts_page'])
train_df, test_df = mean_encoding(train_df, test_df, ['max_counts_page'], self_encoding=False)

# PCA降维
namelist = sorted(list(set(train_beh['page_no'].drop_duplicates().tolist()).intersection(set(test_beh['page_no'].drop_duplicates().tolist()))))
pca_list = []
for name in namelist:
    remove_features.append('page_no_counts_'+str(name))
    pca_list.append('page_no_counts_'+str(name))
pca=PCA(n_components=2)
train_df[pca_list] = train_df[pca_list].fillna(0)
test_df[pca_list] = test_df[pca_list].fillna(0)
pca.fit(train_df[pca_list])
temp_train = pd.DataFrame(pca.transform(train_df[pca_list]))
temp_train.index = train_df.index.tolist()
train_df[['d1', 'd2']] = temp_train
temp_test = pd.DataFrame(pca.transform(test_df[pca_list]))
temp_test.index = test_df.index.tolist()
test_df[['d1', 'd2']] = temp_test

#====================================================================
remove_features.extend(['cur_debit_cnt', 'cur_credit_cnt'])

columns = ['edu_deg_cd', 'mrg_situ_cd', 'acdm_deg_cd', 'deg_cd']
train_df, test_df = mean_encoding(train_df, test_df, columns, self_encoding=False)

#================================导出====================================
train_df.to_pickle('data-minification/train_df.pkl')
test_df.to_pickle('data-minification/test_df.pkl')

remove_features = pd.DataFrame(remove_features, columns=['features_to_remove'])
remove_features.to_pickle('data-minification/remove_features.pkl')