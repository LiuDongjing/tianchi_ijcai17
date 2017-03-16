# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:27:59 2017

@author: LiuYangkai
"""
import logging, argparse
from datarepo import Repo
def main():
    ''''''
    parser = argparse.ArgumentParser(description='控制台调试')
    parser.add_argument('package', help='需要调试的函数所在的包')
    parser.add_argument('function', help='需要调试的函数')
    args = parser.parse_args()
    eval('from %s import %s'%(args.package, args.function))
    Repo()
    while True:
        try:
            eval('%s()'%args.function)
        except Exception as msg:
            logging.warn(msg)
            if len(input('\n按回车重新执行%s.%s()，任意字符终止执行.\n'
                         %(args.package, args.function))) > 0:
                break
            continue
        break

if __name__ == '__main__':
    main()
