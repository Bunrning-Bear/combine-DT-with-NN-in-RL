#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.20
# Modified    :   2017.6.20
# Version     :   1.0


X = [[0., 0.], [1., 1.]]
y = [0, 1]

clf.fit(X,y)
print clf.predict([[2., 2.], [-1., -2.]])