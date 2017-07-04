#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0


import tensorflow as tf 
import numpy as np 
import random
from collections import deque
import logging
# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 3000. # timesteps to observe before training
EXPLORE = 400000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 0.5#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 1024 # size of minibatch
UPDATE_TIME = 100

class BrainDQN(object):
    SIMPLE_OB = 0
    MULTI_OB = 1

    def __init__(self, actions, observations, ob_type,agent_name, **kwargs):
        """
            actions: amount of action
            observations: size of observations
            type:
            """
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.ob_type = ob_type
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        self.observations = observations*10
        self.agent_name = 'saved_networks/' + 'network' + '-dqn-'+agent_name
        # init Q network
        self.network_creator = [
        self.default_multi_layer,
        self.createQNetwork
        ]
        self.stateInput,self.QValue,self.super_para_dic = self.network_creator[ob_type](**kwargs)
        # init Target Q Network
        self.stateInputT,self.QValueT,self.super_para_dic_T = self.network_creator[ob_type](**kwargs)

        # init copy operation
        self.copyTargetQNetworkOperation = []
        for key,value in self.super_para_dic.items():
            assign_op = self.super_para_dic_T[key].assign(self.super_para_dic[key])
            self.copyTargetQNetworkOperation.append(assign_op)
        self.createTrainingMethod()
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            print(checkpoint)
            if checkpoint.model_checkpoint_path.find(self.agent_name) != -1: 
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
        self.merged_summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', self.session.graph)

    def createQNetwork(self,**kwargs):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600,512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512,self.actions])
        b_fc2 = self.bias_variable([self.actions])
        super_para_dic={
        'W_conv1':W_conv1,
        'b_conv1':b_conv1,
        'W_conv2':W_conv2,
        'b_conv2':b_conv2,            
        'W_conv3':W_conv3,
        'b_conv3':b_conv3,    
        'W_fc1':W_fc1,
        'b_fc1':b_fc1,
        'W_fc2':W_fc2,
        'b_fc2':b_fc2,            
        }
        # input layer

        stateInput = tf.placeholder("float",[None,80,80,4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
        return stateInput,QValue,super_para_dic

    def default_multi_layer(self,**kwargs):
        # set x() and y(predict class)
        hidden_layer = kwargs.get('hidden_layer',30) 
        stateInput = tf.placeholder("float",[None, self.observations])

        n_hidden_1 = hidden_layer
        weight = {
            # 随机初始化权重，并且根据层数信息设置相应的初始化维度
            'h1': tf.Variable(tf.random_normal([self.observations, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, hidden_layer])),
            'h3': tf.Variable(tf.random_normal([hidden_layer, hidden_layer])),
            'out': tf.Variable(tf.random_normal([hidden_layer, self.actions]))

            }
        bias = {
        'h1': tf.Variable(tf.random_normal([hidden_layer])),
        'h2': tf.Variable(tf.random_normal([hidden_layer])),
        'h3': tf.Variable(tf.random_normal([hidden_layer])),
        'out': tf.Variable(tf.random_normal([self.actions]))
        }
        # 线性变换
        layer1 = tf.add(tf.matmul(stateInput, weight['h1']), bias['h1'])
        # 带入隐藏层
        layer1 = tf.nn.sigmoid(layer1) 
        out_layer = tf.add(tf.matmul(layer1, weight['out']), bias['out'])
        
        super_para_dic = {
        'w_h1':weight['h1'],
        'w_h2':weight['h2'],
        'w_h3':weight['h3'],
        'w_out':weight['out'],
        'b_h1':bias['h1'],
        'b_h2':bias['h2'],
        'b_h3':bias['h3'],
        'b_out':bias['out']
        }

        return stateInput, out_layer, super_para_dic


    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        tf.scalar_summary('cost', self.cost)  
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        l
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        # Step 2: calculate y 
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))



        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.stateInput : state_batch,
            self.actionInput : action_batch
            })

        # save network every 100000 iteration
        if self.timeStep % 20000 == 0:
            self.saver.save(self.session, self.agent_name, global_step = self.timeStep)

        if self.timeStep % 500 == 0:
            minibatch = random.sample(self.replayMemory,BATCH_SIZE/2)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            nextState_batch = [data[3] for data in minibatch]
            # Step 2: calculate y 
            y_batch = []
            QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
            for i in range(0,BATCH_SIZE/2):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
            
            summary = self.session.run(self.merged_summary_op,feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })  
            self.summary_writer.add_summary(summary, self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            # 把训练的结果赋值给target Q network，用于下次训练用
            self.copyTargetQNetwork()


    def observation_to_state(self,observation):
        if self.ob_type == self.MULTI_OB:
            return np.append(self.currentState[:,:,1:],observation,axis = 2)
        else:
            return np.append(self.currentState[0:self.observations/10*9],observation)

    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = self.observation_to_state(nextObservation)
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE :
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % 100 == 0:
            logging.debug("TIMESTEP", self.timeStep, "/ STATE", state, \
                "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action

    def setInitState(self,observation):
        if self.ob_type == self.MULTI_OB:
            self.currentState = np.stack((observation, observation, observation, observation), axis = 2)
        else:
           self.currentState = np.concatenate((observation, observation, observation, observation, observation, observation, observation, observation, observation, observation), axis = 0)

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

