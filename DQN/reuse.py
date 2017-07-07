#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.7.6
# Modified    :   2017.7.6
# Version     :   1.0



import tensorflow as tf


with tf.variable_scope('scope1'):
    tf.get_variable('v1',[1],initializer=tf.random_uniform_initializer(1,10))
    
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    with tf.variable_scope('scope1',reuse=True):
        print(sess.run(tf.assign(tf.get_variable('v1'), tf.add(tf.get_variable('v1'), [10]))))
with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        with tf.variable_scope('scope1',reuse=True):
            print(sess.run(tf.add(tf.get_variable('v1'), [10])))
