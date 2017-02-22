#-*- coding:utf-8 -*-
#直接用倒数第三个星期的数据预测(2016.10.25-2016.10.31)
import pandas as pd
from time import clock
def main():
    #names指定列名，便于后续的sql查询
    dat = pd.read_csv('../input/dataset/user_pay.txt',
                       header = None, 
                       names = ['uid', 'sid', 'stamp'])
    #select * from dat where stamp >= '' and stamp <= ''
    dat = dat[(dat['stamp'] >= '2016-10-25 00:00:00')
            & (dat['stamp'] <= '2016-10-31 23:59:59')]
    
    #用于后续的left join操作
    day7 = pd.DataFrame([str(e) for e in range(1, 2001)], 
                         columns=['sid'])
    #统计倒数第三个星期的数据
    for k in range(25, 32):
        dat1 = dat[(dat['stamp'] >= ('2016-10-%d 00:00:00' % k))
                & (dat['stamp'] <= ('2016-10-%d 23:59:59' % k))]
        #group by语句，统计每个group的元素个数，注意是Series float类型
        #后续转换成DataFrame才能用join操作
        dat1 = dat1.groupby('sid').size()
        
        #将dat1(Series)的行索引取出来(int list)，也就是商店id，
        #并转换为对应的str list，和dat1的数据组合成一个两列的DataFrame
        sid = []
        for j in range(len(dat1.axes[0])):
            sid.append(str(dat1.axes[0][j]))
        dayi = 'day%d' % (k - 24)
        #注意这种初始化DataFrame的方式
        dat1 = pd.DataFrame({'sid':sid, dayi:dat1.values})
        #左连接，有可能产生np.NaN数据
        day7 = day7.merge(dat1, how = 'left', on = 'sid')
        #update day7 set dayi = 0 where dayi is null
        #将NaN置为0
        day7.loc[day7[dayi].isnull(), dayi] = 0
    last7 = day7
    nmap = {}
    for k in range(1, 8):
        nmap['day%d' % k] = 'day%d' % (k+7)
    #注意这种给列重命名的方式
    last7 = last7.rename_axis(nmap, axis="columns")
    day14 = day7.merge(last7, how = 'left', on = 'sid')
    #指定float_format可只保存float类型数据的整数部分
    day14.to_csv('../temp/baseline.csv', header=False, index=False, encoding='utf-8', 
                  float_format='%0.0f')
#
if __name__ == '__main__':
    clock()
    main()
    print(clock())