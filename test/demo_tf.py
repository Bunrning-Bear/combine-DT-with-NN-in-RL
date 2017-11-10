#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0

# class a(object):
#     test = 10

# if __name__ == '__main__':
#     print a.test
#     dic = {'a':1,'b':2};
#     print dic.get('c',111)
# from demo2 import obj

# class a(object):
#     def __init__(self,obj):
#         self.dict_obj = obj



# test1 = a(obj)
# test1.dict_obj['3']=3
# test2 = a(obj)

# print test2.dict_obj

import tensorflow as tf

a = tf.Variable(-1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print(sess.run(a))

# saver.save(sess, './temp_model/')
saver.restore(sess, './temp_model/')
print(sess.run(a))