# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:46:52 2017

@author: LiuYangkai
"""
import logging
from features import extractAll
from preprocess import process
def main():
    process()
    extractAll()
#
if __name__ == '__main__':
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format   = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt  = '%y%m%d %H:%M:%S',
                    filename = '../temp/log.txt',
                    filemode = 'w');
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    console.setFormatter(logging.Formatter('%(asctime)s %(filename)s: %(levelname)-8s %(message)s'));
    logging.getLogger('').addHandler(console);
    main()
