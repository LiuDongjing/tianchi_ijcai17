# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:46:52 2017

@author: LiuYangkai
"""
import logging, xgboost, os
from features import extractAll
import dataproc
from time import clock
from sklearn.externals import joblib
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from random import randint
def blend(lst, w):
    '''将多个结果根据权重融合起来'''
    r = lst[0] * w[0]
    for k in range(1, len(lst)):
        r = r + lst[k]*w[k]
    return r.round()
def select_test(n, count):
    '''从大小为n的样本中随机选择count个作为测试，其余的用来训练'''
    index = [False for _ in range(n)]
    p = 0
    while p < count:
        ii = randint(0, n-1)
        if index[ii]:
            continue
        index[ii] = True
        p += 1
    return index
def official_loss(estimator, X, y):
    #注意重置index，不然会出现意想不到的问题
    y_ = y.reset_index(drop=True)
    y_p = estimator.predict(X)
    adds = (y_p + y_).abs()
    subs = (y_p - y_).abs()
    divs = subs / adds
    N = divs.shape[0] * divs.shape[1]
    return divs.sum().sum() / N
class WarpModel(BaseEstimator):
    ''''''
    def __init__(self, model):
        klass = model.__class__
        self.modelList = []
        for k in range(14):
            self.modelList.append(klass(**(model.get_params(deep=False))))
        self.__initParams = {}
        self.__initParams['model'] = model
    def get_params(self, deep=False):
        '''返回构造该类的参数, 因为交叉验证的函数会clone传进去的model对象，会调用该方法
        '''
        return self.__initParams
    def fit(self, X, y):
        '''X=[n_samples, n_features]
           y=[n_samples, n_targets]'''
        #注意重置index，不然模型计算过程中会出错
        X_ = X.reset_index(drop=True)
        for k in range(14):
            xt = X_.iloc[:,0:14]
            xt.insert(14, 'maxt', X_.iloc[:, 14+2*k])
            xt.insert(15, 'desc', X_.iloc[:, 14+2*k+1])
            y_ = y.iloc[:, k].reset_index(drop=True)
            self.modelList[k].fit(xt, y_)
        return self
    def predict(self, X):
        '''X=[n_samples, n_features]'''
        #注意重置index，不然模型计算过程中会出错
        dat = X.reset_index(drop=True)
        labels = pd.DataFrame()
        for k in range(14):
            xt = dat.iloc[:,0:14]
            xt.insert(14, 'maxt', dat.iloc[:,14+2*k])
            xt.insert(15, 'desc', dat.iloc[:,14+2*k+1])
            p = self.modelList[k].predict(xt)
            p = pd.DataFrame({'tmp%d'%k:p})
            labels.insert(k, 'day%d'%(k+1), p)
        labels.columns = ['day%d'%k for k in range(1, 15)]
        return labels.round()#array, (n_samples,)
def main():
    model = None
    #mode = 'predict'
    modelPath = '../temp/model.pkl'
    modelWeight = [0.3, 0.4, 0.3]
    modelName = ['xgboost', 'GBDT', 'RandomForest']
    if os.path.exists(modelPath):
        logging.info('从%s中加载模型...'%modelPath)
        #joblib.load加载的是保存到磁盘中的对象，不仅仅是训练好的模型
        modelList = joblib.load(modelPath) 
        feature = extractAll('predict')
        X = feature.iloc[:, 1:]
        resList = []
        for k in range(len(modelName)):
            model = modelList[k]
            logging.info('%s:预测中...'%modelName[k])
            resList.append(model.predict(X))
            logging.info('%s:预测结束.'%modelName[k])
        logging.info('融合模型...')
        y = blend(resList, modelWeight)
        y.insert(0, 'sid', feature['sid'])
        y.to_csv('../temp/result.csv', header=False, index=False,
                 encoding='utf-8', float_format='%0.0f')
        logging.info('已将结果保存到../temp/result.csv')
    else:
        dataproc.preprocess()
        (feature, label) = extractAll()
        logging.info('共有%d条训练数据.' % feature.shape[0])
        index1 = feature['day1'] > 0
        index2 = label['day1'] > 0
        for k in range(2, 15):
            index1 = index1 | feature['day%d'%k] > 0
            index2 = index2 | label['day%d'%k] > 0
        index = index1 & index2
        feature = feature[index]
        label = label[index]
        logging.info('去掉无效数据后还剩%d条.' % (feature.shape[0]))
        test_set = select_test(feature.shape[0], round(feature.shape[0]*0.1))
        test_feature = feature.loc[test_set, :]
        test_label = label.loc[test_set, :]
        logging.info('用%d个样本用作最终的测试.'%test_feature.shape[0])
        for k in range(len(test_set)):
            test_set[k] = not test_set[k]
        feature = feature.loc[test_set, :]
        label = label.loc[test_set, :]
        logging.info('用%d个样本用作训练.'%feature.shape[0])
        modelList = [WarpModel(xgboost.XGBRegressor(
                          silent=True, n_estimators=100)), 
                     WarpModel(GradientBoostingRegressor()),
                     WarpModel(RandomForestRegressor())]
        
        for k in range(len(modelList)):
            logging.info('%s: 交叉验证...'%modelName[k])
            model = modelList[k]
            #注意n_jobs使用多CPU时，不可以调试，否则会抛出异常
            scores = cross_val_score(model, feature, label, 
                                     cv=KFold(n_splits=3, shuffle=False), 
                                     #n_jobs=-1, 
                                     scoring=official_loss
                                     )
            logging.info('交叉验证结果：%s' % str(scores))
            logging.info('用所有的训练数据训练模型...')
            model.fit(feature, label)
            modelList[k] = model
            logging.info('%s测试模型...' % modelName[k])
            logging.info('测试结果: %f' % official_loss(model, test_feature, \
                                                    test_label))
        joblib.dump(modelList, modelPath)
        logging.info('已将训练好的模型保存到%s.'%modelPath)
#
if __name__ == '__main__':
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format   = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt  = '%y%m%d %H:%M:%S',
                    filename = '../temp/log.txt',
                    filemode = 'w');
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(logging.Formatter('%(asctime)s %(filename)s: %(levelname)-8s %(message)s'));
    logging.getLogger('').addHandler(console);
    clock()
    main()
    logging.info('共耗时%f分钟.' % (clock()/60))
