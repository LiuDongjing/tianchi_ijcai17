# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:24:40 2017

@author: dxy
"""
## INSTRUCTION:
##  1. Put 'user_pay.txt' in the same file with this script
##  2. The result will be saved in the current file, namely 'average.csv'

import pandas as pd
import numpy as np

##############################################################################
## statistic about how many day of week from 2015-7-1 to 2016-10-31 
##                                                expect 2015-12-12
start_date=pd.datetime(2015,7,1)
end_date=pd.datetime(2016,10,31)
index=pd.date_range(start_date,end_date)
dayofweek=pd.Series(index.dayofweek,index=index)
dayofweek=dayofweek.drop(pd.datetime(2015,12,12))
dayofweek=dayofweek.groupby(dayofweek).size()
# answer:
#   0(Mon)  1(Tues)  2(Wed)  3(Thur)  4(Fri)  5(Sat)  6(Sun)
#   70      69       70      70       70      69      70

##############################################################################
#fileloc='../../data/dataset/'

# Open user_pay.txt and 
# add a column with value from 0-6 representing the day of week
fileloc=''
filename='../input/dataset/user_pay.txt'
user_pay=pd.read_csv(fileloc+filename,sep=',',header=None)
user_pay.columns=['user_id','shop_id','time_stamp']
user_pay['time_stamp'] = pd.to_datetime(user_pay['time_stamp'])
user_pay['day_of_week'] = user_pay['time_stamp'].dt.dayofweek

##############################################################################
# For a fixed store, id=1 for example, calculate all purchasing times happened 
# in Monday, then divide how many Mondays in this period of time and finally, 
# put the value into a column called purchase_time.
# (The algorithm is same for Thuesday to Sunday) 
user_pay_grouped=user_pay.groupby(['shop_id','day_of_week']).size()

user_pay_grouped=user_pay_grouped.div(dayofweek,level=1) # Series type
user_pay_grouped=user_pay_grouped.to_frame()
user_pay_grouped=user_pay_grouped.reset_index()
user_pay_grouped=user_pay_grouped.rename(columns = {0:'purchase_times'})

##############################################################################
# Generate the final table
ds=pd.pivot_table(user_pay_grouped,values='purchase_times', 
                   index='shop_id', columns='day_of_week',
                   fill_value=0)
ds=ds.round().astype(np.int)

start=(end_date.weekday()+1)%7
result = pd.concat([ds,ds,ds],axis=1,ignore_index=True)
result = result.loc[:,start:start+13]

##############################################################################
# save result
path='../temp/average.csv'
result.to_csv(path,sep=',',header=False,index=True,encoding='utf-8')
#user_pay.groupby(['shop_id', 
#                  user_pay['time_stamp'].dt.date]).size()










