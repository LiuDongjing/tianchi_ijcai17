# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:41:23 2017

@author: LiuYangkai
"""
from time import clock
import logging, os, re, time
import pandas as pd
class Repo:
    '''设计成单例模式'''
    __ins = None
    def __init__(self, baseDir='../temp/repo'):
        self.dir = baseDir
        self.data = {}
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            logging.info('新建文件夹: %s'%self.dir)
        for p in os.listdir(self.dir):
            if os.path.isfile(
                    os.path.join(self.dir, p)):
                key = re.split(r'.', p)[0]
                path = os.path.join(self.dir, p)
                t = pd.read_pickle(path)
                logging.info('从%s中加载%s.'%(path, key))
                self.data[key] = t
    def __new__(cls, *args, **kwargs):
        '''在__init__之前调用'''
        if not cls.__ins:
            cls.__ins = super(Repo, cls).__new__(
                    cls, *args, **kwargs)
        return cls.__ins
    def __call__(self, func, *args, **kwargs):
        '''重写了ins(func, args)方法.
        这里的func是个可调用的对象'''
        if not callable(func):
            raise Exception('%s不是可调用的对象!' % str(func))
        key = getattr(func, '__name__')
        if key in self.data:
            #返回数据的副本
            return self.data[key].copy(deep=True)
        clock()
        t = func(*args, **kwargs)
        dur = clock()
        #计算时间超过20s就把结果缓存下来
        if dur >= 20:
            path = os.path.join(self.dir, key + '.pkl')
            t.to_pickle(path)
            logging.info('已将%s缓存到%s.' % (key, path))
        self.data[key] = t
        #返回数据的副本
        return t.copy(deep=True)
    def saveResult(self, result, name='none'):
        stamp = time.strftime('%m%d_%H:%M',time.localtime(time.time()))
        path = os.path.join(self.dir, '%s_%s.csv'%(stamp, name))
        result.to_csv(path, index=False, encoding='utf-8')