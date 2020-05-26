# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:11:17 2020

@author: mache
"""


import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings("ignore")


NFOLDS = 8
testingModel = False

#=================================================================
train_dt2 = pd.read_pickle('eda/train_df.pkl')
test_dt2 = pd.read_pickle('eda/test_df.pkl')
remove_features2 = pd.read_pickle('eda/remove_features.pkl')
remove_features2 = list(remove_features2['features_to_remove'].values)

#=================================================================
# remove_features2.append('trx_days')
remove_features2.extend(['trx_counts_5', 'trx_counts_6'])
remove_features2.append('real_sum_trx_amt')
#=================================================================
remove_features2.extend(['l12mon_buy_fin_mng_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms'])
remove_features2.extend(['ic_ind', 'deg_cd', 'ovd_30d_loan_tot_cnt', 'l6mon_agn_ind', 'hav_hou_grp_ind'])
remove_features2.append('job_year')


remove_features2.append('trx_sum_night')
remove_features2.extend(['page_tm_hour_dis'])

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
                remove_features2.append(new_col_name)
                

X_train2 = train_dt2.drop(remove_features2, axis=1)
y_train2 = train_dt2['flag']

remove_features2.remove('flag')
X_test2 = test_dt2.drop(remove_features2, axis=1)

#=================================================================
remove_features3 = pd.read_pickle('eda/remove_features.pkl')
remove_features3 = list(remove_features3['features_to_remove'].values)
remove_features3.extend(['mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd'])

X_train3 = train_dt2.drop(remove_features3, axis=1)
y_train3 = train_dt2['flag']

remove_features3.remove('flag')
X_test3 = test_dt2.drop(remove_features3, axis=1)

    
#=================================================================

if testingModel:
    
    lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.005,
                    'n_estimators':5500,
                    'num_leaves': 28,
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
    
    X, X_predict, y, y_predict = train_test_split(X_train2, y_train2, test_size=0.1, shuffle=True, random_state=666)
    
    ntrain = X.shape[0]
    ntest = X_predict.shape[0]
    kf = KFold(n_splits=NFOLDS, random_state=2055)
    
    def get_oof(X_train, y_train, X_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            kf_X_train = X_train.iloc[train_index, :]
            kf_y_train = y_train[train_index]
            kf_X_test = X_train.iloc[test_index, :]
            
            tr_data = lgb.Dataset(kf_X_train, kf_y_train)
            estimator = lgb.train(lgb_params,
                                  tr_data,
                                  verbose_eval=200
                                  )
            
            oof_train[test_index] = estimator.predict(kf_X_test)
            oof_test_skf[i, :] = estimator.predict(X_test)
        
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    oof_train3, oof_test3 = get_oof(X, y, X_predict)
    print('lgb-CV AUC is', metrics.roc_auc_score(y_predict, oof_test3))
    
    #-----------------------------------------------------------
    params={
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth':5,
            'lambda':10,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'min_child_weight':2,
            'eta': 0.025,
            'seed':0,
            'nthread':8,
            'silent':1
            }
    
    X, X_predict, y, y_predict = train_test_split(X_train3, y_train3, test_size=0.33, shuffle=True, random_state=666)
    
    ntrain = X.shape[0]
    ntest = X_predict.shape[0]
    kf = KFold(n_splits=NFOLDS, random_state=2017)
    
    def get_oof(X_train, y_train, X_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            kf_X_train = X_train.iloc[train_index, :]
            kf_y_train = y_train[train_index]
            kf_X_test = X_train.iloc[test_index, :]
            
            dtrain = xgb.DMatrix(kf_X_train, label=kf_y_train)
            dtest = xgb.DMatrix(X_test)
            dtest2 = xgb.DMatrix(kf_X_test)
            
            watchlist = [(dtrain,'train')]
            model=xgb.train(params,dtrain,num_boost_round=600, evals=watchlist, verbose_eval=False)
            
            oof_train[test_index] = model.predict(dtest2)
            oof_test_skf[i, :] = model.predict(dtest)
        
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    # oof_train4, oof_test4 = get_oof(X, y, X_predict)
    # print('xgb-CV AUC is', metrics.roc_auc_score(y_predict, oof_test4))
    
    # #-----------------------------------------------------------
    # oof_train = np.hstack((oof_train3, oof_train4))
    # oof_test = np.hstack((oof_test3, oof_test4))

    # model = LinearRegression()
    # model.fit(oof_train, y)
    # print('系数矩阵:\n',model.coef_)
    # print('截距项:\n',model.intercept_)
    # pre_y = model.predict(oof_test)   
    
    # print('Stacking AUC is', metrics.roc_auc_score(y_predict, pre_y))

else:    
   
    lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.005,
                    'n_estimators':5500,
                    'num_leaves': 28,
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
    
    X, X_predict, y = X_train2, X_test2,  y_train2
    
    ntrain = X.shape[0]
    ntest = X_predict.shape[0]
    kf = KFold(n_splits=NFOLDS, random_state=880)
    
    def get_oof(X_train, y_train, X_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            kf_X_train = X_train.iloc[train_index, :]
            kf_y_train = y_train[train_index]
            kf_X_test = X_train.iloc[test_index, :]
            
            tr_data = lgb.Dataset(kf_X_train, kf_y_train)
            estimator = lgb.train(lgb_params,
                                  tr_data,
                                  verbose_eval=200
                                  )
            
            oof_train[test_index] = estimator.predict(kf_X_test)
            oof_test_skf[i, :] = estimator.predict(X_test)
        
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    oof_train3, oof_test3 = get_oof(X, y, X_predict)
    
    #-----------------------------------------------------------
    params={
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth':5,
            'lambda':10,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'min_child_weight':2,
            'eta': 0.025,
            'seed':0,
            'nthread':8,
            'silent':1
            }
    
    X, X_predict, y = X_train3, X_test3,  y_train3
    
    ntrain = X.shape[0]
    ntest = X_predict.shape[0]
    kf = KFold(n_splits=NFOLDS, random_state=2017)
    
    def get_oof(X_train, y_train, X_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
        
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            kf_X_train = X_train.iloc[train_index, :]
            kf_y_train = y_train[train_index]
            kf_X_test = X_train.iloc[test_index, :]
            
            dtrain = xgb.DMatrix(kf_X_train, label=kf_y_train)
            dtest = xgb.DMatrix(X_test)
            dtest2 = xgb.DMatrix(kf_X_test)
            
            watchlist = [(dtrain,'train')]
            model=xgb.train(params,dtrain,num_boost_round=600, evals=watchlist, verbose_eval=False)
            
            oof_train[test_index] = model.predict(dtest2)
            oof_test_skf[i, :] = model.predict(dtest)
        
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
    # oof_train4, oof_test4 = get_oof(X, y, X_predict)
    
    # #-----------------------------------------------------------
    # oof_train = np.hstack((oof_train3, oof_train4))
    # oof_test = np.hstack((oof_test3, oof_test4))
    
    # model = LinearRegression()
    # model.fit(oof_train, y)
    # print('系数矩阵:\n',model.coef_)
    # print('截距项:\n',model.intercept_)
    # pre_y = model.predict(oof_test)
    
    test_dt2['prediction'] = oof_test3
    with open('stacking_res/prediction_lgb.txt', 'w+') as f:
        for i in range(len(test_dt2)):
            f.write(test_dt2['id'][i])
            f.write('\t')
            f.write(str(test_dt2['prediction'][i]))
            f.write('\n')
    
    s = open('stacking_res/prediction_lgb.txt', mode='r', encoding='utf-8-sig').read()
    open('stacking_res/prediction_lgb.txt', mode='w', encoding='utf-8').write(s)
