#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0
import re
GAME_NAME = 'AirRaid-ram-v0'
FOREST_SIZE = 10
MAX_DEPTH = 5
MAX_VALUE = 9999999999

# Data format names.
ATTR_TYPE_NOMINAL = NOM = 'nominal'
ATTR_TYPE_DISCRETE = DIS = 'discrete'
ATTR_TYPE_CONTINUOUS = CON = 'continuous'
ATTR_MODE_CLASS = CLS = 'class'
ATTR_HEADER_PATTERN = re.compile("([^,:]+):(nominal|discrete|continuous)(?::(class))?")
REWARD = 'reward'
ACTION = 'action'
TERMINAL = 'terminal'
STATE_ATTRS = [REWARD, ACTION, TERMINAL]

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0.02#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 1024 # size of minibatch
UPDATE_TIME = 100

# baisc_model_name_prefix
MODEL_NAME = 'basic_qnetwork'
MODEL_PATH ='saved_networks/'