# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:06:08 2017

@author: LiuYangkai
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/dataset/user_pay.txt', header=None,
                 names=['uid', 'sid', 'stamp'], dtype=np.str)
#df = df[(df['stamp'] >= '2015-11-08 00:00:00') &
#        (df['stamp'] <= '2015-11-14 23:59:59')]
df['stamp'] = df['stamp'].str[:10]
gb = df.groupby('stamp').size()
gb = gb.values
gb = gb / 2000
plt.plot(gb)
plt.show()
