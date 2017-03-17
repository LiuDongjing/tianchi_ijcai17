# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:27:59 2017

@author: LiuYangkai
    写这段脚本是为了方便调试程序，因为跑算法的代码通常需要加载上百兆的数据，而且计算很多
    中间结果比较耗时，所以这段脚本将需要加载的数据和中间结果缓存到内存，并且可以重复调用
    待调试的代码，这样在反复调试的过程中就可以显著减少加载数据和计算中间结果的时间。
"""
import logging, argparse
from datarepo import Repo
def main():
    '''有两个命令行参数 package和function，function指定待调试的函数，而package指定
    该函数在哪个包。每次执行完，都可以选择继续执行，或者终止执行'''
    parser = argparse.ArgumentParser(description='用于调试package.function。\
                                     每次执行完，都可以选择继续执行，或者终止执行')
    parser.add_argument('package', help='需要调试的函数所在的包')
    parser.add_argument('function', help='需要调试的函数')
    args = parser.parse_args()
    #导入待调试的函数
    eval('from %s import %s'%(args.package, args.function))
    #初始化数据管理对象
    Repo()
    while True:
        try:
            eval('%s()'%args.function)
        except Exception as msg:
            logging.warn(msg)
            pass
        if len(input('\n按回车重新执行%s.%s()，任意字符终止执行.\n'
                     %(args.package, args.function))) > 0:
            break

if __name__ == '__main__':
    main()
