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
from sklearn.linear_model import LinearRegression
def official_loss(estimator, X, y):
    #注意重置index，不然会出现意想不到的问题
    y_ = y.reset_index(drop=True)
    y_p = estimator.predict(X)
    adds = (y_p + y_).abs()
    subs = (y_p - y_).abs()
    divs = subs / adds
    N = divs.shape[0] * divs.shape[1]
    return divs.sum().sum() / N
class Warpxgboost(BaseEstimator):
    ''''''
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        '''X=[n_samples, n_features]
           y=[n_samples, n_targets]'''
        #注意重置index，不然模型计算过程中会出错
        X_ = X.reset_index(drop=True)
        y_ = y.iloc[:, 0].reset_index(drop=True)
        self.model.fit(X_, y_)
        return self
    def predict(self, X):
        '''X=[n_samples, n_features]'''
        #注意重置index，不然模型计算过程中会出错
        dat = X.reset_index(drop=True)
        labels = pd.DataFrame()
        for k in range(14):
            p = self.model.predict(dat)
            p = pd.DataFrame({'tmp%d'%k:p})
            labels.insert(k, 'day%d'%(k+1), p)
            dat = dat.drop(dat.columns[0], axis='columns')
            dat.insert(len(dat.columns), '_%d'%k, p)
            dat.columns = ['day%d'%n for n in range(1, 8)]
        labels.columns = ['day%d'%k for k in range(1, 15)]
        return labels#array, (n_samples,)
def main():
    model = None
    if os.path.exists('../temp/model.pkl'):
        model = joblib.load('../temp/model.pkl') 
    else:
        dataproc.preprocess()
        (feature, label) = extractAll()
        logging.info('共有%d条训练数据.' % feature.shape[0])
        index = (feature['day1'] > 0) |\
                (feature['day2'] > 0) |\
                (feature['day3'] > 0) |\
                (feature['day4'] > 0) |\
                (feature['day5'] > 0) |\
                (feature['day6'] > 0) |\
                (feature['day7'] > 0)
        for k in range(14):
            index = index | label['day%d'%(k+1)] > 0
        feature = feature[index]
        label = label[index]
        logging.info('去掉无效数据后还剩%d条.' % (feature.shape[0]))
        model = xgboost.XGBRegressor(silent=True,  n_estimators=100)
        #model = LinearRegression()
        logging.info('开始交叉验证...')
        #注意n_jobs使用多CPU时，不可以调试，否则会抛出异常
        scores = cross_val_score(Warpxgboost(model), feature, label, 
                                 cv=KFold(n_splits=3, shuffle=True), 
                                 n_jobs=-1, 
                                 scoring=official_loss)
        logging.info('交叉验证的结果%s' % str(scores))
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
    clock()
