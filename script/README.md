# 部分脚本功能说明
- [main.py](main.py)。基于xgboost的混合模型的代码。
- [features.py](features.py)。提取特征和标签的代码。
- [tensorflow.py](tensorflow.py)。神经网络模型的代码。
- [head.py](head.py)。工具函数，可抽取部分数据用于调试程序。
	```
		usage: head.py [-h] [--number NUMBER] [--random] [--count] file

		查看文件的前N行或者随机N行，也可以统计文件的行数

		positional arguments:
		  file                  输入文件

		optional arguments:
		  -h, --help            show this help message and exit
		  --number NUMBER, -n NUMBER
								行数
		  --random, -r          是否启用随机模式
		  --count, -c           统计文件有多少行
	```
- [datarepo.py](datarepo.py)。用于管理占用空间大和计算比较耗时的数据。里面的Repo类设计成单例模式，方便在不同脚本之间统一管理数据。使用方法很简单：
	```python
        rep = Repo()
        dat = rep(lambda x:x, pd.DataFrame(np.random.randn(4, 4))
	```
    首先获取Repo对象，然后把该对象当做函数来调用，注意第一个参数是个可执行的函数，用于计算待存储的数据，后面的参数都是该可执行函数的参数。Repo对象会将计算结果缓存起来，如果计算时间超过20秒，还会将计算结果保存到辅存。
- [console.py](console.py)。写这段脚本是为了方便调试程序，因为跑算法的代码通常需要加载上百兆的数据，而且计算很多中间结果比较耗时，所以这段脚本结合_datarepo.py_将需要加载的数据和中间结果缓存到内存，并且可以重复调用待调试的代码，这样在反复调试的过程中就可以显著减少加载数据和计算中间结果的时间。
	```
		usage: console.py [-h] package function

		用于调试package.function。 每次执行完，都可以选择继续执行，或者终止执行

		positional arguments:
		  package     需要调试的函数所在的包
		  function    需要调试的函数

		optional arguments:
		  -h, --help  show this help message and exit
	```