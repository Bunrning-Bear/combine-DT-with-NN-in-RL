#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.22
# Modified    :   2017.6.22
# Version     :   1.0
import tensorflow as tf
# from dtree import Tree, Data, USE_NEAREST,unique,Forest
import numpy as np
import random
import csv
import logging
from base_data_structure import *
import time
logging.basicConfig(level=logging.INFO)

class Basic_model(object):

    def __init__(self,freatures_amount,class_amount,global_flag, 
        learning_rate=0.01,min_learning_rate=0.00001,tol=0.00001,max_iter=40,
        multi_layer=None,hidden_layer=150,
        # batch_size=300
        ):

        self.freatures_amount = freatures_amount
        self.class_amount = class_amount
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        # self.batch_size = batch_size
        self.model_name = global_flag+'-'+str(self.freatures_amount) + '-' + str(self.class_amount)
        self.tol=tol
        self.max_iter = max_iter
        self.weight = {}
        self.bias = {}
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder('float',[None,self.freatures_amount])
            self.y = tf.placeholder('float', [None, self.class_amount])
            self.Y = tf.placeholder('float',[None,self.class_amount])
            self.pred = self.default_multi_layer(hidden_layer) if multi_layer == None else multi_layer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
            self.init = tf.initialize_all_variables()
            self.predict_single = tf.argmax(self.pred, 1)
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
            # self.saver = tf.train.Saver()
            self.g.finalize()
        self.sess = tf.Session(graph=self.g)
        self.incremental_times = 0
        self.last_weight = []
        # 可以在一个类里面初始化一个会话的示例，供其他方法调用，而不是每一个方法开启一个会话然后再关闭


    def _onehot(self,labels):
        ''' one-hot 编码 '''
        n_sample = len(labels)# sample 的个数
        # n_class = max(labels) + 1 # 最大类别个数，label从0开始算
        onehot_labels = np.zeros((n_sample, self.class_amount))# 构造二维数组
        onehot_labels[np.arange(n_sample), labels] = 1 
        # 第一个维度使用np.arange(n_sample) 来获得sample的下标；
        # 第二个维度指定label所在的位置为1
        return onehot_labels

    def default_multi_layer(self,hidden_layer):
        # set x() and y(predict class)

        n_hidden_1 = hidden_layer
        self.weight = {
            # 随机初始化权重，并且根据层数信息设置相应的初始化维度
            'h1': tf.Variable(tf.random_normal([self.freatures_amount, n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, self.class_amount]))

        }
        self.bias = {
            'h1': tf.Variable(tf.random_normal([n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.class_amount]))

        }
        # 线性变换
        layer1 = tf.add(tf.matmul(self.x, self.weight['h1']), self.bias['h1'])
        # 带入隐藏层
        layer1 = tf.nn.sigmoid(layer1) 
        out_layer = tf.add(tf.matmul(layer1, self.weight['out']), self.bias['out'])
        return out_layer

    def train(self,X,Y,initial=False,display_step=False):
        """Initial model or incremental training model.
        
        Args:
            X: train data, numpy type
            Y: label data, numpy type

        """
        # with self.g.as_default():
        # with tf.Session() as self.sess:

        if initial:
            # 若要进行增量训练，则将initial 设置为false。只有第一次训练的时候需要进行参数的初始化
            # 初始化需要在计算图构造完成后进行，不能在初始化完成后进行新增的计算图
            
            self.sess.run(self.init)
            print ("complete initial")
        else:
            # 先进行初始化，然后再载入模型！
            # self.sess.run(tf.initialize_all_variables())
            # ckp = self.saver.last_checkpoints[-1]
            # # logging.info("load model, checkpoints %s"%ckp)
            # # self.saver = tf.train.import_meta_graph(''.join([ckp,'.meta']))
            # # logging.info("load model, checkpoints %s"%self.saver.last_checkpoints)
            # self.saver.restore(self.sess, ckp)
            # # logging.info("restore finfish")
            # self.last_weight = self.sess.run(self.weight['h1'])
            pass

        n_sample = X.shape[0]
        Y = self._onehot(Y)
        last_cost = 999999999999999999999999999
        # total_batch = int(n_sample / self.batch_size)
        # logging.info("total batch is %s"%total_batch)
        for epoch in range(self.max_iter):
            _, current_cost = self.sess.run([self.optimizer,self.cost],
                feed_dict={self.x:X,
                           self.y:Y})
            # 分批次进行迭代训练
            
            # avg_cost = 0.

            # for i in range(total_batch):
            #     # note:  dot sumbol usage??
            #     _, c = self.sess.run([self.optimizer,self.cost],
            #         feed_dict={self.x:X[i*self.batch_size : (i+1)*self.batch_size , :],
            #                    self.y:Y[i*self.batch_size : (i+1)*self.batch_size , :]})
            #     avg_cost += c/total_batch
            
            if display_step:
                logging.info("Epoch: %s, Cost = %s"%(epoch+1,current_cost))

            if(last_cost - current_cost < self.tol):
                if display_step:
                    logging.info("cost decrease less than %s,break"%(self.tol))
                break
                # adaptive learning rate
                # learning_rate = learning_rate / 5
                # if self.min_learning_rate > learning_rate:
                #     # logging.info("learning rate is small than %s, break epoch"%self.min_learning_rate)
                #     break
            last_cost = current_cost
        logging.info('Opitimization Finished!')
        this_weight = self.sess.run(self.weight['h1'])
        # if self.last_weight !=[]:
        #     a = tf.equal(self.last_weight, this_weight)
        #     print self.sess.run(a)
        # print self.saver.save(self.sess, self.model_name)
        # print self.saver.last_checkpoints
        self.incremental_times+=1

    def predict(self,X):
        # with self.g.as_default():
        #     with tf.Session() as self.sess:
                # with tf.Session() as sess:
        # self.saver.restore(self.sess, self.saver.last_checkpoints[0])
        predict = self.sess.run(self.predict_single,feed_dict={self.x:X})
        return predict

    def test(self,X,Y):
        # with tf.Session() as self.sess:
        # with self.g.as_default():
        #     with tf.Session() as self.sess:
        Y = self._onehot(Y)
        # self.saver.restore(self.sess, self.saver.last_checkpoints[0])  
        acc = self.sess.run(self.accuracy,feed_dict={self.x:X,self.y:Y})
        return acc


def main():
    train_data = Data('dataset/uci_adult/delete_adult2.data')
    class_name = 'cls'
    print "listing..."
    ori = list(train_data)
    random.shuffle(ori)
    res_data = []
    res_label = []
    size = len(ori)
    incre_times = 10
    batch_size = 4000
    for list_item in ori:
        sample_item = []
        for key,value in list_item.items():
            if key == class_name:
                res_label.append(value)
            else:
                sample_item.append(value)
        res_data.append(sample_item)
    res_data = np.array(res_data)
    res_label = np.array(res_label)


    test = Data('dataset/uci_adult/delete_adult2.test')
    ori = list(test)
    random.shuffle(ori)
    res_data_test = []
    res_label_test = []
    for list_item in ori:
        sample_item = []
        for key,value in list_item.items():
            if key == class_name:
                res_label_test.append(value)
            else:
                sample_item.append(value)
        res_data_test.append(sample_item)
    res_data_test = np.array(res_data_test)
    res_label_test = np.array(res_label_test)

    freatures_amount = res_data.shape[1]
    model_list = []
    for i in range(10):
        start = time.clock()
        logging.info("initial model %s"%i)
        model_list.append(Basic_model(freatures_amount, 2 , global_flag='base_model_para'))
        end = time.clock()
        print "time used %s"%(end-start)

    for model in model_list:
        start = time.clock()
        model.train(res_data[:batch_size], res_label[:batch_size],initial=True,display_step=True)
        end = time.clock()
        print "time used of training: %s"%(end-start)
        pre =  model.predict(res_data[0].reshape(1,14)) 
        start = time.clock()
        end = time.clock()
        print ("predict is %s true value is %s"%(pre,res_label[0]))
        print "time used of predict: %s"%(end-start)
        start = time.clock()       
        print model.test(res_data_test,res_label_test)
        end = time.clock()
        print "time used of test: %s"%(end-start)        
        start = time.clock()             
        for i in range(5):
            print "incremental learning times %s"%i
            model.train(res_data[batch_size*(i+1):batch_size*(i+2)], res_label[batch_size*(i+1):batch_size*(i+2)],initial=False,display_step=True)
            print model.test(res_data_test,res_label_test)
        end = time.clock()
        print "time used of incremental learning: %s"%(end-start)      

if __name__ == '__main__':
    main()