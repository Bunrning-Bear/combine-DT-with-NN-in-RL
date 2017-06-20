#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.9.9
# Modified    :   2017.9.9
# Version     :   1.0



from dtree import Tree, Data, USE_NEAREST,unique
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import csv
import logging

train_data = Data('cdata6')
# train_data = Data('dataset/mnist/new_strain.csv')
test_data = Data('cdata5')
# print train_data.attribute_names
# print "attribute test %s",%train_data.get_attribute_type('pixel317')
# uni = unique([record['pixel317'] for record in train_data])
# print str(min(uni))+"   "+str(max(uni))



class_name = 'cls'
print "listing..."
ori = list(train_data)
# print ori
# # ori = ori[:len(ori)/3]
# print "suffuling..."
# random.shuffle(ori)
# print "reshaping..."
# res_data = []
# res_label = []
# for list_item in ori:
#     sample_item = []
#     for key,value in list_item.items():
#         if key == class_name:
#             res_label.append(value)
#         else:
#             sample_item.append(value)
#     res_data.append(sample_item)
# train_data_transformed = zip(res_data,res_label)
# print train_data_transformed
# # print res_data
# # print res_label
# size = len(res_data)
# print "data size is %s"%size


# Tree.build(train_data)
tree = Tree.build(train_data)
tree.set_missing_value_policy(USE_NEAREST)
print "----------------------------"
for item in ori:
    logging.info("distributing: %s"%item)
    Tree.distribute(tree,item)

print "----------------------------"
tree.incremental_training_Driver()
prediction = tree.predict({'a':1,'b':1,'c':1,'d':3})

right_amount = 0
for item in test_data:
    pre = tree.predict(item)
    if pre[0] ==item[class_name]:
        right_amount = right_amount + 1
print float(right_amount)/len(test_data)
# result = tree.test(test_data)
# print 'Accuracy:',result.mean
# prediction = tree.predict({'a':1,'b':1,'c':1,'d':3})
# print 'best:',prediction.best
# print 'probs:',prediction.probs

# tree = Tree.build(Data('regression-training.csv'))
# result = t.test(Data('regression-testing.csv'))
# print 'MAE:',result.mean
# prediction = tree.predict(dict(feature1=123, feature2='abc', feature3='hot'))
# print 'mean:',prediction.mean
# print 'variance:',prediction.variance

# print "using warm_start = True"
# clf = MLPClassifier(hidden_layer_sizes=(250,), max_iter=30, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.2,learning_rate='adaptive', warm_start=True)
# print "fitting..."
# split = size*3/5
# clf.fit(res_data[split:],res_label[split:])
# right_amount = 0
# for item in zip(res_data,res_label):
#     sample = np.array(item[0])
#     pre = clf.predict(sample.reshape(1,-1))
#     if pre[0] ==item[1]:
#         right_amount = right_amount + 1
# print float(right_amount)/len(train_data)
# print "fitting2..."
# clf.fit(res_data[:split],res_label[:split])

# right_amount = 0
# for item in zip(res_data,res_label):
#     sample = np.array(item[0])
#     pre = clf.predict(sample.reshape(1,-1))
#     if pre[0] ==item[1]:
#         right_amount = right_amount + 1
# print float(right_amount)/len(train_data)
# '''
# 1:0.111976190476
# 2:0.111976190476
# '''

# '''
# 1:0.111523809524
# 2:0.111523809524

# '''
# print "using warm_start = False"
# clf = MLPClassifier(hidden_layer_sizes=(250,), max_iter=30, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.2,learning_rate='adaptive',warm_start=False)
# print "fitting..."
# clf.fit(res_data[split:],res_label[split:])
# right_amount = 0
# for item in zip(res_data,res_label):
#     sample = np.array(item[0])
#     pre = clf.predict(sample.reshape(1,-1))
#     if pre[0] ==item[1]:
#         right_amount = right_amount + 1
# print float(right_amount)/len(train_data)
# print "fitting2..."
# clf.fit(res_data[:split],res_label[:split])

# right_amount = 0
# for item in zip(res_data,res_label):
#     sample = np.array(item[0])
#     pre = clf.predict(sample.reshape(1,-1))
#     if pre[0] ==item[1]:
#         right_amount = right_amount + 1
# print float(right_amount)/len(train_data)

# ite = csv.reader(open('cdata4'))
# # print [row for row in ite]
# print [i for i in train_data]