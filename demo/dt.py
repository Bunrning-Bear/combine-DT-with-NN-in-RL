#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.5.11
# Modified    :   2017.5.11
# Version     :   1.0
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO

iris = load_iris()
print iris.data[2,[0,2]]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data,iris.target)
# # with open("iris.dot", 'w') as f:
# #     f = tree.export_graphviz(clf, out_file=f)
# print clf.predict_proba(iris.data[:1, :])