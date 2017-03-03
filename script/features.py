# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:20:03 2017

@author: LiuYangkai
"""
import os, logging
import pandas as pd
import numpy as np
def crossJoin(df1, df2):
    '''两个DataFrame的笛卡尔积'''
    df1['_temp_key'] = 1
    df2['_temp_key'] = 1
    df = df1.merge(df2, on='_temp_key')
    df = df.drop('_temp_key', axis='columns')
    return df
def getLabels(dat, mode='train'):
    '''获取未来14天的销量'''
    dat = dat[(dat['stamp'] >= '2015-07-01') & 
              (dat['stamp'] <= '2016-10-31')]
    dat['stamp'] = dat['stamp'].str[:10]
    dat = dat.groupby('sid')
    days = None
    for sid in [str(k) for k in range(1, 2001)]:
        shop = dat.get_group(sid)
        logging.info('getLabels:%s.' % sid)
        shop = shop.drop('sid', axis='columns')
        shop = shop.groupby('stamp').size().reset_index()
        shop.rename_axis({0:'sales'}, axis='columns', inplace=True)
        shop = shop.sort_values('stamp')
        N = shop.shape[0]
        if N < 14:
            logging.warn('%s的数据条数不足14个.'%sid)
            continue
        full = pd.DataFrame({'stamp':
                    pd.date_range(shop['stamp'][0], '2016-10-31').\
                                 strftime('%Y-%m-%d')})
        shop = full.merge(shop, how='left', on='stamp')
        shop.fillna(0, inplace=True)
        idx = shop[shop['stamp'] == '2015-12-12'].axes[0]
        if len(idx) >= 1:
            if idx[0] > 0:
                shop.loc[idx, 'sales'] = round((shop.loc[idx-1, 'sales'].\
                        values[0] + shop.loc[idx+1, 'sales'].values[0])/2)
            else:
                shop.loc[idx, 'sales'] = shop.loc[idx+1, 'sales'].values[0]
        #前7天用于提取特征
        temp = pd.DataFrame({'stamp':shop['stamp'][7:-14].reset_index(drop=True)})
        N = shop.shape[0]
        for n in range(14):
            t = shop['sales'][7+n:N+n-14].reset_index(drop=True)
            temp.insert(n+1, 'day%d'%(n+1), t)
        temp['sid'] = sid
        if days is None:
            days = temp
        else:
            days = days.append(temp)
    return days
def extractAll(mode = 'train'):
    featurePath = os.path.join('../temp/', mode + '_features.csv')
    labelPath = os.path.join('../temp/', mode + '_labels.csv')

    if os.path.exists(featurePath) and\
       os.path.exists(labelPath):
        return (pd.read_csv(featurePath), 
                pd.read_csv(labelPath))

    #提取特征
    logging.info('加载user_pay.txt...')
    user_pay = pd.read_csv('../input/dataset/user_pay.txt', 
                           header = None, names = ['uid', 'sid', 'stamp'],
                           dtype = np.str)
    logging.info('提取Label...')
    labels = getLabels(user_pay)
    f1 = Last_week_sales(mode=mode)
    logging.info('提取最近7天的销量数据...')
    f1 = f1.extract(user_pay)
    features = f1
    if not (features['sid'].equals(labels['sid'])) or\
       not (features['stamp'].equals(labels['stamp'])):
        features.to_csv(featurePath+'.dump', index=False, encoding='utf-8')
        labels.to_csv(labelPath+'.dump', index=False, encoding='utf-8')
        raise Exception('特征和标签不匹配！数据已保存到dump。')
    #保存计算的features到outPath
    features = features.drop(['sid', 'stamp'], axis='columns')
    labels = labels.drop(['sid', 'stamp'], axis='columns')
    logging.info('保存提取的特征和label...')
    features.to_csv(featurePath, index=False, encoding='utf-8')
    labels.to_csv(labelPath, index=False, encoding='utf-8')
    
    return (features, labels)
#
class BaseFeature:
    def __init__(self, outDir = '../temp/', 
                 featureName = 'base', mode = 'train'):
        self.outFile = os.path.join(outDir, mode + '_' + featureName + '.csv')
        self.name = featureName
        self.data = None
        if os.path.exists(self.outFile):
            self.data = pd.read_csv(self.outFile, dtype = np.str)
            logging.info('从%s中载入特征%s.' % (self.outFile, self.name))
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
        dat = indata
        dat['stamp'] = dat['stamp'].str[:10]
        dat = dat[(dat['stamp'] >= '2015-07-01') & 
          (dat['stamp'] <= '2016-10-17')]
        dat = dat.groupby('sid')
        days = None
        for sid in [str(k) for k in range(1, 2001)]:
            shop = dat.get_group(sid)
            logging.info('last_week_sales:%s.' % sid)
            shop = shop.drop('sid', axis='columns')
            shop = shop.groupby('stamp').size().reset_index()
            shop.rename_axis({0:'sales'}, axis='columns', inplace=True)
            shop = shop.sort_values('stamp')
            N = shop.shape[0]
            if N < 7:
                logging.warn('%s的数据条数不足7个.'%sid)
                continue
            full = pd.DataFrame({'stamp':
                        pd.date_range(shop['stamp'][0], '2016-10-17').\
                                     strftime('%Y-%m-%d')})
            shop = full.merge(shop, how='left', on='stamp')
            shop.fillna(0, inplace=True)
            idx = shop[shop['stamp'] == '2015-12-12'].axes[0]
            if len(idx) >= 1:
                if idx[0] > 0:
                    shop.loc[idx, 'sales'] = round((shop.loc[idx-1, 'sales'].\
                            values[0] + shop.loc[idx+1, 'sales'].values[0])/2)
                else:
                    shop.loc[idx, 'sales'] = shop.loc[idx+1, 'sales'].values[0]
            temp = pd.DataFrame({'stamp':shop['stamp'][7:].reset_index(drop=True)})
            N = shop.shape[0]
            for n in range(7):
                t = shop['sales'][n:N-7+n].reset_index(drop=True)
                temp.insert(n+1, 'day%d'%(n+1), t)
            temp['sid'] = sid
            if days is None:
                days = temp
            else:
                days = days.append(temp)
        days.to_csv(self.outFile, index=False, encoding='utf-8', 
                          float_format='%0.0f')
        logging.info('已将最近7天的销售数据保存到%s.'%self.outFile)
        return days
