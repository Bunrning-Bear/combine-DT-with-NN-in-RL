#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0

class a(object):
    test = 10

if __name__ == '__main__':
    print a.test
    dic = {'a':1,'b':2};
    print dic.get('c',111)