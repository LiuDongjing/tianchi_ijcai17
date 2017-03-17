# IJCAI-17 口碑商家客流量预测解题代码

## 问题描述
> 在这次比赛中，每只队伍需要预测测试集中所有商家在未来14天
> （2016.11.01-2016.11.14）内各自每天（00:00:00-23:59:59）的
> 客户流量。预测结果为非负整数。

天池平台提供2015-07-01到2016-10-31(除去2015-12-12)的商家数据、
用户支付行为数据以及用户浏览行为数据，要求参赛选手预测2000家商户在2016-11-01-2016-11-14的客户流量。

Loss函数的定义如下：
![L = \frac{1}{nT}\sum_i^n \sum_t^T\left|\frac{c_{it} - c_{it}^g}{c_{it} + c_{it}^g}\right|](http://mathurl.com/kctnfdv.png)

其中![c_{it}](http://mathurl.com/l3le8ue.png)是第t天商家i的客户流量预测值，![c_{it}^g](http://mathurl.com/k7ryzbr.png)是第t天商家i的客户流量实际值。

## 解题思路
### 构造数据集
针对每个商户(sid)的每一天(stamp)可以提取一个样本，具体做法是把sid在stamp-1至stamp-14的销售量等数据作为特征，而在stamp至stamp+14的销售量作为对应的标签。

### 特征及标签
> sid,stamp,day1,day2,...,day14,maxt1,desc1,maxt2,desc2,...,maxt14,desc14

特征的数据组织形式。sid是商家的id，stamp是日期yyyy-mm-dd，day1-day14是stamp前14天的销量(不包括stamp)，maxt1-maxt14是未来14天的最高温度(包括stamp)，desc1-desc14表示未来14天是否下雨(包括stamp)。

> sid,stamp,day1,day2,...,day14

标签的数据组织形式。sid和stamp同上，day1-day14是未来14天的销量，其中day1即是stamp当天的销量。

当验证完特征的sid以及stamp和标签的一一对应后，就会移除两者的sid和stamp域，组成最终的数据集。

### 模型
#### 基于xgboost的混合模型
用WarpModel包装基本的回归模型(比如xgboost)使之可以输出14天的预测值，基本思想是在WarpModel里包含14个基本的模型的对象，每个对象负责预测某一天的销量，14个模型的训练和预测过程相互独立。

在WarpModel的基础上分别使用xgboost、GBDT和RandomForest作为它的基本模型，得到三个结果，赋予相应的权重获得最终结果。

#### 三层神经网络
特征、标签和xgboost方法的一样，不过只使用了前14天的销量。每一天的销售量用一个三层的子网络训练得到，14个子网络分别迭代训练若干次后，再用官方的loss函数训练整个网络。

### 结果
基于xgboost的混合模型效果要明显好于神经网络的方法，提交结果后的loss值是**0.092552**，而神经网络线下测试的loss是**0.623244**，效果太差就没提交上去:joy:。