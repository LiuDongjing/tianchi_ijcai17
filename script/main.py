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
def main():
    model = None
    if os.path.exists('../temp/model.pkl'):
        model = joblib.load('../temp/model.pkl') 
    else:
        dataproc.preprocess()
        (feature, label) = extractAll()
        logging.info('共有%d条训练数据.' % feature.shape[0])
        index = (feature['day1'] != 0) |\
                (feature['day2'] != 0) |\
                (feature['day3'] != 0) |\
                (feature['day4'] != 0) |\
                (feature['day5'] != 0) |\
                (feature['day6'] != 0) |\
                (feature['day7'] != 0)
        feature = feature[index]
        label = label[index]
        logging.info('去掉无效数据后还剩%d条.' % (feature.shape[0]))
        logging.info('开始交叉验证...')
        model = xgboost.XGBRegressor(silent=True)
        scores = cross_val_score(model, feature, label, 
                                 cv=KFold(n_splits=3), 
                                 n_jobs=-1)
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
