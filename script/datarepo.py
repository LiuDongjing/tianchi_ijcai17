# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 20:41:23 2017

@author: LiuYangkai
"""
from time import clock
import logging, os, re, time
import pandas as pd
class Repo:
    '''用于管理占用空间大和计算比较耗时的数据。该类设计成单例模式，方便在不同脚本之间
    统一管理数据。使用方法很简单：
        rep = Repo()
        dat = rep(lambda x:x, pd.DataFrame(np.random.randn(4, 4))
    首先获取Repo对象，然后把该对象当做函数来调用，注意第一个参数是个可执行的函数，用于
    计算待存储的数据，后面的参数都是该可执行函数的参数。Repo对象会将计算结果缓存起来，
    如果计算时间超过20秒，还会将计算结果保存到辅存。
    '''
    
    __ins = None #单例模式下用于保存那个唯一的对象
    def __init__(self, baseDir='../temp/repo'):
        '''baseDir是缓存数据的文件夹'''
        self.dir = baseDir
        self.data = {}
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            logging.info('新建文件夹: %s'%self.dir)
        
        #如果缓存文件夹里有数据，就将它们加载到内存
        for p in os.listdir(self.dir):
            if os.path.isfile(
                    os.path.join(self.dir, p)):
                key = re.split(r'.', p)[0]
                path = os.path.join(self.dir, p)
                t = pd.read_pickle(path)
                logging.info('从%s中加载%s.'%(path, key))
                self.data[key] = t
    def __new__(cls, *args, **kwargs):
        '''在__init__之前调用，保证每次调用构造的时候，返回的都是同一个对象'''
        if not cls.__ins:
            cls.__ins = super(Repo, cls).__new__(
                    cls, *args, **kwargs)
        return cls.__ins
    def __call__(self, func, *args, **kwargs):
        '''重写了ins(func, args, kwargs)方法.
        这里的func是个可调用的对象，其余的是func的参数。'''
        if not callable(func):
            raise Exception('%s不是可调用的对象!' % str(func))
        #用函数名作为数据的名称
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
        '''用于管理最终的结果，每次保存结果会加上时间信息，方便跟踪结果的改进过程'''
        stamp = time.strftime('%m%d_%H:%M',time.localtime(time.time()))
        path = os.path.join(self.dir, '%s_%s_result.csv'%(stamp, name))
        result.to_csv(path, index=False, encoding='utf-8')
