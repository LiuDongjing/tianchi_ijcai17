#coding=utf-8
import numpy as np
import pandas as pd

# your path to table user_pay
user_pay = 'user_pay.txt'

# load data
print('loading data...')
user_pay_df = pd.read_table(user_pay, sep=',', header=None, \
    names=['user_id', 'shop_id', 'time_stamp'], \
    dtype={'user_id':'str', 'shop_id':'str', 'time_stamp':'str'})

# generate customer flow
print('generating customer flow...')
user_pay_df['time_stamp'] = user_pay_df['time_stamp'].str[:10]
customer_flow = user_pay_df.groupby(['shop_id', 'time_stamp']).size()
# predict
fid = open('prediction_example.csv', 'w')
for shop_id in xrange(1, 2001):
    print('predicting: %4d/2000'%shop_id)
    weekly_flow = pd.Series(np.zeros(7, dtype=int), 
        [d.strftime('%Y-%m-%d') for d in pd.date_range('10/25/2016', periods=7)])
    flow = customer_flow.loc[str(shop_id), '2016-10-25':'2016-10-31']
    weekly_flow[flow.index.get_level_values(1)] = flow
    # use latest week's customer flow to predict following 2 weeks' customer flow
    predictons = ','.join([str(x) for x in list(weekly_flow)*2])
    fid.write('%d,%s\n'%(shop_id, predictons))
fid.close()
print('Finish')