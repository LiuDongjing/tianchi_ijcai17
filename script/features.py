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
    cache_path = '../temp/label_with_sid_stamp.csv'
    if os.path.exists(cache_path):
        logging.info('从%s读取数据.'%cache_path)
        dtype={'sid':np.str, 'stamp':np.str}
        for k in range(14):
            dtype['day%d'%(k+1)] = np.int
        return pd.read_csv(cache_path,
                           dtype=dtype)
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
        #前14天用于提取特征
        temp = pd.DataFrame({'stamp':shop['stamp'][14:-14].reset_index(drop=True)})
        N = shop.shape[0]
        for n in range(14):
            t = shop['sales'][14+n:N+n-14].reset_index(drop=True)
            temp.insert(n+1, 'day%d'%(n+1), t)
        temp['sid'] = sid
        if days is None:
            days = temp
        else:
            days = days.append(temp)
    days.to_csv(cache_path, index=None)
    return days
def extractAll(mode = 'train'):
    featurePath = os.path.join('../temp/', mode + '_features.csv')
    labelPath = os.path.join('../temp/', mode + '_labels.csv')
    labels = None
    if mode == 'train':
        if os.path.exists(featurePath) and\
           os.path.exists(labelPath):
            return (pd.read_csv(featurePath), 
                    pd.read_csv(labelPath))
    else:
        if os.path.exists(featurePath):
            return pd.read_csv(featurePath)
    #提取特征
    logging.info('加载user_pay.txt...')
    user_pay = pd.read_csv('../input/dataset/user_pay.txt', 
                           header = None, names = ['uid', 'sid', 'stamp'],
                           dtype = np.str)
    
    if mode == 'train' and labels is None:
        logging.info('提取Label...')
        labels = getLabels(user_pay)
    f1 = Last_week_sales(mode=mode)
    logging.info('提取最近14天的销量数据...')
    f1 = f1.extract(user_pay)
    f2 = Weather(mode=mode)
    logging.info('提取天气数据...')
    f2 = f2.extract()
    if mode == 'train':
        features = f1.merge(f2, on=['sid','stamp'], how='left')
    else:
        features = f1.merge(f2, on=['sid'], how='left')
    if features.isnull().any().any():
        raise Exception('存在无效数据!')
    features = features.reset_index(drop=True)
    if mode == 'train':
        labels = labels.reset_index(drop=True)
    if mode == 'train':
        if not (features['sid'].equals(labels['sid'])) or\
           not (features['stamp'].equals(labels['stamp'])):
            features.to_csv(featurePath+'.dump', index=False, encoding='utf-8')
            labels.to_csv(labelPath+'.dump', index=False, encoding='utf-8')
            raise Exception('特征和标签不匹配！数据已保存到dump。')
        #保存计算的features到outPath
        features = features.drop(['sid', 'stamp'], axis='columns')
        labels = labels.drop(['sid', 'stamp'], axis='columns')
        logging.info('保存提取的label...')
        labels.to_csv(labelPath, index=False, encoding='utf-8')
    logging.info('保存提取的特征...')
    features.to_csv(featurePath, index=False, encoding='utf-8')
    if mode == 'train':
        return (features, labels)
    return features
#
class BaseFeature:
    def __init__(self, outDir = '../temp/', 
                 featureName = 'base', mode = 'train',
                 dtype=np.str):
        self.outFile = os.path.join(outDir, mode + '_' + featureName + '.csv')
        self.name = featureName
        self.mode = mode
        self.data = None
        if os.path.exists(self.outFile):
            self.data = pd.read_csv(self.outFile, dtype = dtype)
            logging.info('从%s中载入特征%s.' % (self.outFile, self.name))
    def extract(self, indata):
        return self.data
#
class Last_week_sales(BaseFeature):
    def __init__(self, mode = 'train'):
        dtype = {'sid':np.str, 'stamp':np.str}
        for k in range(1, 15):
            dtype['day%d'%k] = np.int
        BaseFeature.__init__(self, 
                             featureName = 'Last_two_weeks_sales',
                             mode = mode,
                             dtype=dtype)
    def extract(self, indata):
        if self.data is not None:
            return self.data
        if isinstance(indata, str):
            indata = pd.read_csv(indata, header = None)
        #提取特征
        dat = indata
        dat['stamp'] = dat['stamp'].str[:10]
        if self.mode == 'train':
            dat = dat[(dat['stamp'] >= '2015-07-01') & 
                      (dat['stamp'] <= '2016-10-18')]
        else:
            dat = dat[(dat['stamp'] >= '2016-10-18') & 
                      (dat['stamp'] <= '2016-10-31')]
        if self.mode != 'train':
            dat = dat.groupby('sid')
            cols = ['sid']
            for k in range(14):
                cols.append('day%d'%(k+1))
            days = pd.DataFrame(columns=cols)
            for sid in [str(k) for k in range(1, 2001)]:
                tmp = {}
                tmp['sid'] = sid
                try:
                    sale = dat.get_group(sid)
                    sale = sale.groupby('stamp').size()
                    for k in range(14):
                        try:
                            tmp['day%d'%(k+1)] = sale.loc['2016-10-%d'%(18+k)]
                        except:
                            tmp['day%d'%(k+1)] = 0
                except:
                    logging.warn('%s在提取特征时间段没有销售量'%sid)
                    for k in range(1, 15):
                        tmp['day%d'%k] = 0
                days = days.append(tmp, ignore_index=True)
            days.to_csv(self.outFile, index=False, encoding='utf-8', 
                          float_format='%0.0f')
            logging.info('已将最近14天的销售数据保存到%s.'%self.outFile)
            return days
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
            if N < 14:
                logging.warn('%s的数据条数不足14个.'%sid)
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
            temp = pd.DataFrame({'stamp':shop['stamp'][14:].reset_index(drop=True)})
            N = shop.shape[0]
            for n in range(14):
                t = shop['sales'][n:N-14+n].reset_index(drop=True)
                temp.insert(n+1, 'day%d'%(n+1), t)
            temp['sid'] = sid
            if days is None:
                days = temp
            else:
                days = days.append(temp)
        days.to_csv(self.outFile, index=False, encoding='utf-8', 
                          float_format='%0.0f')
        logging.info('已将最近14天的销售数据保存到%s.'%self.outFile)
        return days
class Weather(BaseFeature):
    '''提取每家店所在城市的天气数据，雨天为True，否则False；
        还包括了最高温度'''
    def __init__(self, mode='train'):
        dtype = {'sid':np.str, 'stamp':np.str}
        for k in range(1, 15):
            dtype['maxt%d'%k] = np.float
            dtype['desc%d'%k] = np.bool
        BaseFeature.__init__(self, 
                             featureName = 'weather',
                             mode = mode, dtype=dtype)
    def extract(self, indata=None):
        if self.data is not None:
            return self.data
        logging.info('读取文件%s.'%'../input/weather_all.csv')
        wh = pd.read_csv('../input/weather_all.csv', 
                         header=None, names=['city', 
                         'stamp', 'maxt', 'desc'],
                         usecols=[0,1,2,4], dtype={'city':np.str,
                                 'stamp':np.str,'maxt':np.int,'desc':np.str})
        if self.mode == 'train':
            wh = wh[(wh['stamp'] >= '2015-07-15') &
                    (wh['stamp'] <= '2016-10-31')]
        else:
            wh = wh[(wh['stamp'] >= '2016-11-01') &
                    (wh['stamp'] <= '2016-11-14')]
        wh.loc[:, 'desc'] = wh.desc.apply(lambda s:'雨' in s)
        logging.info('读取文件%s.'%'../input/dataset/shop_info.txt')
        shop_info = pd.read_csv('../input/dataset/shop_info.txt',
                                header=None, names=['sid', 'city'],
                                usecols=[0,1], dtype=np.str)
        weather = shop_info.merge(wh, on='city', how='left')
        weather = weather.drop('city', axis='columns')
        gb = weather.groupby('sid')
        weather = None
        if self.mode != 'train':
            for sid in [str(e) for e in range(1, 2001)]:
                logging.info('weather:%s.'%sid)
                print(sid)
                dat = gb.get_group(sid).reset_index(drop=True)
                dat = dat.sort_values('stamp')
                dat = dat.drop('stamp', axis='columns')
                tmp = pd.DataFrame({'sid':[sid]})
                for k in range(14):
                    tmp.insert(2*k+1, 'maxt%d'%(k+1),dat.loc[k, 'maxt'])
                    tmp.insert(2*k+2, 'desc%d'%(k+1),dat.loc[k, 'desc'])
                if weather is None:
                    weather = tmp
                else:
                    weather = weather.append(tmp)
            weather.to_csv(self.outFile, index=False, encoding='utf-8')
            logging.info('已将天气特征保存到%s.'%self.outFile)
            return weather
        for sid in [str(e) for e in range(1, 2001)]:
            logging.info('weather:%s.'%sid)
            print(sid)
            dat = gb.get_group(sid).reset_index(drop=True)
            dat = dat.drop('sid', axis='columns')
            dat = dat.sort_values('stamp')
            tmp = pd.DataFrame({'stamp':
                        pd.date_range('2015-07-15', '2016-10-18').\
                                     strftime('%Y-%m-%d')})
            N = dat.shape[0]
            for k in range(14):
                 tmp.insert(2*k+1, 'maxt%d'%(k+1), dat.loc[k:N-14+k, 'maxt']\
                            .reset_index(drop=True))
                 tmp.insert(2*k+2, 'desc%d'%(k+1), dat.loc[k:N-14+k, 'desc']\
                            .reset_index(drop=True))
            tmp['sid'] = sid
            if weather is None:
                weather = tmp
            else:
                weather = weather.append(tmp)
        weather.to_csv(self.outFile, index=False, encoding='utf-8')
        logging.info('已将天气特征保存到%s.'%self.outFile)
        return weather
        