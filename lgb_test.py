# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:51:17 2020

@author: mache
"""


import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings("ignore")
 

NFOLDS = 5

df = pd.read_pickle('eda/train_df.pkl')
remove_features = pd.read_pickle('eda/remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)


# df['Dat_Flg1_Cd_counts_B'].fillna(0)
# df['Dat_Flg1_Cd_sum_amt_B'].fillna(0)
# df['l12'] = df.apply(lambda x: (x['l12mon_buy_fin_mng_whl_tms'] + 
#                                 x['l12_mon_insu_buy_whl_tms'] + x['l12_mon_fnd_buy_whl_tms'] +
#                                 x['l12_mon_gld_buy_whl_tms']) / (x['Dat_Flg1_Cd_counts_B']+1e-2), axis=1)
# df['l12'] = df.apply(lambda x: (x['l12mon_buy_fin_mng_whl_tms'] + 
#                                 x['l12_mon_insu_buy_whl_tms'] + x['l12_mon_fnd_buy_whl_tms'] +
#                                 x['l12_mon_gld_buy_whl_tms']) / (x['Dat_Flg1_Cd_sum_amt_B']+1e-2), axis=1)

# df['B/C'] = df.apply(lambda x: x['Dat_Flg1_Cd_sum_amt_B'] / x['Dat_Flg1_Cd_sum_amt_C'] 
#                      if (x['Dat_Flg1_Cd_sum_amt_C'] != np.nan and x['Dat_Flg1_Cd_sum_amt_C'] > 0) 
#                      else np.nan, axis=1)

# remove_features.extend(['d2'])

remove_features.append('trx_days2')

remove_features.extend(['trx_counts_5', 'trx_counts_6'])
remove_features.append('real_sum_trx_amt')
# remove_features.append('trx_counts')
#####################################################################
# remove_features.extend(['l12mon_buy_fin_mng_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms'])
remove_features.extend(['ic_ind', 'deg_cd', 'ovd_30d_loan_tot_cnt', 'l6mon_agn_ind', 'hav_hou_grp_ind'])
remove_features.append('job_year')

# remove_features.extend(['d1'])

# remove_features.append('trx_sum_night')
remove_features.extend(['page_tm_hour_dis'])

main_columns = ['real_sum_trx_amt', 
          'l12mon_buy_fin_mng_whl_tms',
          'trx_counts'
          ]
uids = ['l1y_crd_card_csm_amt_dlm_cd', 'perm_crd_lmt_cd'
        ]
aggregations = ['mean']
for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = col+'_'+main_column+'_'+agg_type
                remove_features.append(new_col_name)


# main_columns= ['ovd_30d_loan_tot_cnt', 'his_lng_ovd_day'
#           ]
# uids = ['loan_act_ind', 'pl_crd_lmt_cd'
#         ]
# for main_column in main_columns:  
#         for col in uids:
#             new_col_name = col+'_'+main_column+'_'+'dis'
#             remove_features.append(new_col_name)
#-------------------------------------------------------------------    
X = df.drop(remove_features, axis=1)
y = df['flag']

pearsonArr = X.corr()
####################################################################
#==============================调参===================================
#-------------------------确定n_estimators---------------------------

# params_test0={'n_estimators':range(101, 143, 2)}
              
# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.1,
#                     'tree_learner':'serial'
#                 }

# grid_list = []
# for k in params_test0.keys():
#     for val in params_test0[k]:
#         print(val)
#         params[k] = val
        
#         folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
#         predictions = 0
#         for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#             # print('Fold:', fold_)
#             tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#             vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
            
#             # print(len(tr_x), len(vl_x))
#             tr_data = lgb.Dataset(tr_x, label=tr_y)
#             estimator = lgb.train(params,
#                                   tr_data,
#                                   verbose_eval=200
#                                   )
#             pre_y = estimator.predict(vl_x)
#             predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
        
#         par_list = dict()
#         for k2 in params_test0.keys():
#             par_list[k2] = params[k2]
#         grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))

# #-------------------------确定num_leaves---------------------------

# params_test1={'num_leaves':[28,30, 31, 32, 33, 34,36, 38]}
              
# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 2**5,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':1.0,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.001,
#                     'lambda_l2':0.46,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 } 

# grid_list = []
# for k in params_test1.keys():
#     for val in params_test1[k]:
#         params[k] = val
#         print(val)
#         folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=56)
#         predictions = 0
#         for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#             # print('Fold:', fold_)
#             tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#             vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
            
#             # print(len(tr_x), len(vl_x))
#             tr_data = lgb.Dataset(tr_x, label=tr_y)
#             estimator = lgb.train(params,
#                                   tr_data,
#                                   verbose_eval=200
#                                   )
#             pre_y = estimator.predict(vl_x)
#             predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
        
#         par_list = dict()
#         for k2 in params_test1.keys():
#             par_list[k2] = params[k2]
#         grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))


# #--------------------确定 min_data_in_leaf 和 max_bin---------------------
# params_test2={'max_bin': [118,119,120], 'min_data_in_leaf':[46,47,48]}
  
# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 2**5,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':1.0,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.01,
#                     'lambda_l2':0.5,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 }        

# grid_list = []
# for v1 in params_test2['max_bin']:
#     for v2 in params_test2['min_data_in_leaf']:
#         print(v1, v2)
#         params['max_bin'] = v1
#         params['min_data_in_leaf'] = v2
        
#         folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
#         predictions = 0
#         for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#             # print('Fold:', fold_)
#             tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#             vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
            
#             # print(len(tr_x), len(vl_x))
#             tr_data = lgb.Dataset(tr_x, label=tr_y)
#             estimator = lgb.train(params,
#                                   tr_data,
#                                   verbose_eval=200
#                                   )
#             pre_y = estimator.predict(vl_x)
#             predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
        
#         par_list = dict()
#         for k2 in params_test2.keys():
#             par_list[k2] = params[k2]
#         grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))


# #------------确定 feature_fraction、bagging_fraction、bagging_freq----------
# params_test3={'feature_fraction': [1],
#               'bagging_fraction': [0.6, 0.7, 0.8, 0.9],
#               'bagging_freq': [0, 10, 20]
# }

# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 2**5,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':1.0,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.01,
#                     'lambda_l2':0.5,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 } 

# grid_list = []
# for v1 in params_test3['feature_fraction']:
#     for v2 in params_test3['bagging_fraction']:
#         for v3 in params_test3['bagging_freq']:
#             print(v1, v2, v3)
#             params['feature_fraction'] = v1
#             params['bagging_fraction'] = v2
#             params['bagging_freq'] = v3
            
#             folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
#             predictions = 0
#             for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#                 # print('Fold:', fold_)
#                 tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#                 vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
            
#                 # print(len(tr_x), len(vl_x))
#                 tr_data = lgb.Dataset(tr_x, label=tr_y)
#                 estimator = lgb.train(params,
#                                       tr_data,
#                                       verbose_eval=200
#                                       )
#                 pre_y = estimator.predict(vl_x)
#                 predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
            
#             par_list = dict()
#             for k2 in params_test3.keys():
#                 par_list[k2] = params[k2]
#             grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))

# #-----------------------确定 lambda_l1 和 lambda_l2-------------------------
# params_test4={'lambda_l1': [0.001],
#               'lambda_l2': [0.4, 0.42, 0.44, 0.45, 0.46]
# }

# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 2**5,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':1.0,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.001,
#                     'lambda_l2':0.46,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 } 

# grid_list = []
# for v1 in params_test4['lambda_l1']:
#     for v2 in params_test4['lambda_l2']:
#         print(v1, v2)
#         params['lambda_l1'] = v1
#         params['lambda_l2'] = v2
        
#         folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
#         predictions = 0
#         for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#             # print('Fold:', fold_)
#             tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#             vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
        
#             # print(len(tr_x), len(vl_x))
#             tr_data = lgb.Dataset(tr_x, label=tr_y)
#             estimator = lgb.train(params,
#                                   tr_data,
#                                   verbose_eval=200
#                                   )
#             pre_y = estimator.predict(vl_x)
#             predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
        
#         par_list = dict()
#         for k2 in params_test4.keys():
#             par_list[k2] = params[k2]
#         grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))


# #-----------------------确定 min_split_gain-------------------------
# params_test5={'min_split_gain':[0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 27,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':0.99,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.01,
#                     'lambda_l2':0.5,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 } 

# grid_list = []
# for v1 in params_test5['min_split_gain']:
#     print(v1)
#     params['min_split_gain'] = v1
    
#     folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
#     predictions = 0
#     for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#         # print('Fold:', fold_)
#         tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#         vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
    
#         # print(len(tr_x), len(vl_x))
#         tr_data = lgb.Dataset(tr_x, label=tr_y)
#         estimator = lgb.train(params,
#                               tr_data,
#                               verbose_eval=200
#                               )
#         pre_y = estimator.predict(vl_x)
#         predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
    
#     par_list = dict()
#     for k2 in params_test5.keys():
#         par_list[k2] = params[k2]
#     grid_list.append((predictions, par_list))
# print(max(grid_list, key=lambda x: x[0]))
    
# #--------------------------------验证模型--------------------------------
# params = {
#                     'objective':'binary',
#                     'boosting_type':'gbdt',
#                     'metric':'auc',
#                     'n_jobs':-1,
#                     'learning_rate':0.01,
#                     'n_estimators':1000,
#                     'num_leaves': 32,
#                     'tree_learner':'serial',
#                     'max_bin':119,
#                     'min_data_in_leaf':47,
#                     'feature_fraction':1,
#                     'bagging_fraction':0.6,
#                     'bagging_freq':0,
#                     'lambda_l1':0.01,
#                     'lambda_l2':0.5,
#                     'min_split_gain':0,
#                     'feature_fraction_seed':2,
#                     'bagging_seed':3
#                 }   
# folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=56)
# predictions = 0
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
#     # print('Fold:', fold_)
#     tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
#     vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

#     # print(len(tr_x), len(vl_x))
#     tr_data = lgb.Dataset(tr_x, label=tr_y)
#     estimator = lgb.train(params,
#                           tr_data,
#                           verbose_eval=200
#                           )
#     pre_y = estimator.predict(vl_x)
#     predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
# print("调参后的模型auc:", predictions)

########################################################################
#===========================最佳模型================================
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'n_estimators':1000,
                    'num_leaves': 2**5,
                    'tree_learner':'serial',
                    'max_bin':119,
                    'min_data_in_leaf':47,
                    'feature_fraction':1.0,
                    'bagging_fraction':0.6,
                    'bagging_freq':0,
                    'lambda_l1':0.001,
                    'lambda_l2':0.46,
                    'min_split_gain':0,
                    'feature_fraction_seed':2,
                    'bagging_seed':3
                } 

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=6)
predictions = 0
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:', fold_)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

    # print(len(tr_x), len(vl_x))
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    estimator = lgb.train(lgb_params,
                          tr_data,
                          verbose_eval=200
                          )
    pre_y = estimator.predict(vl_x)
    print(metrics.roc_auc_score(vl_y, pre_y))
    predictions += metrics.roc_auc_score(vl_y, pre_y) / NFOLDS
print("最终模型auc:", predictions)

feature_importance = pd.DataFrame({
        'column': X.columns.tolist(),
        'importance': estimator.feature_importance(importance_type='gain'),
    }).sort_values(by='importance')
