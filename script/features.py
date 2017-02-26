# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:20:03 2017

@author: LiuYangkai
"""
import os, logging
import pandas as pd
def extractAll(mode = 'train'):
    outPath = os.path.join('../temp/', mode + '_features.csv')
    if os.path.exists(outPath):
        return pd.read_csv(outPath, header = None)
    features = pd.read_csv('../input/dataset/user_pay.txt', 
                           header = None, names = ['uid', 'sid', 'stamp'])
    features['stamp'] = features['stamp'].str[:10]
    
    #保存计算的features到outPath
    features.to_csv(outPath, header=False, index=False, encoding='utf-8')
#
class BaseFeature:
    def __init__(self, outDir = '../temp/', 
                 featureName = 'base', mode = 'train'):
        self.outFile = os.path.join(outDir, mode + '_' + featureName + '.csv')
        self.name = featureName
        self.data = None
        if os.path.exists(self.outFile):
            self.data = pd.read_csv(self.outFile, header = None)
            logging.info('从%s中载入特征%s.' (self.outFile, self.name))
    def extract(self, indata):
        return self.data
#
class Last_week_sales(BaseFeature):
    def __init__(self, mode = 'train'):
        BaseFeature.__init__(self, 
                             featureName = 'Last_week_sales',
                             mode = mode)
    def extract(self, indata):
        if self.data is not None:
            return self.data
        if isinstance(indata, str):
            indata = pd.read_csv(indata, header = None)
        #提取特征
        return indata
