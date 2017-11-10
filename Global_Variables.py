#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0
import re
# GAME_NAME = 'CartPole-v0'
# FOREST_SIZE = 2
# MAX_DEPTH = 2
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
OBSERVE = 1000 # timesteps to observe before training
FINAL_EPSILON = 0.01#0.001 # final value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember

# baisc_model_name_prefix
MODEL_PREFIX_PATH = 'saved_networks/'
RECORD_PREFIX_PATH ='record/'
VIDEO_PREFIX_PATH = 'video/'
MODEL_NAME = 'basic_qnetwork'
# MODEL_PATH ='saved_networks_%s_forest_%s_depth/'%(str(FOREST_SIZE),str(MAX_DEPTH))

# network parameter
CPU_NUM = 4
UPDATE_TARGET_INTERVAL= 5000
SAVE_MODEL_INTERVAL= 200000
SCHEDULE_TIMES = 20000

# task
REWARD_GOAL = 200
PRECISION = 1000

# decistion
class TimeStepHolder(object):
    def __init__(self,time):
        self._timesteps = time
    
    def set_time(self,time):
        self._timesteps = time

    def inc_time(self):
        self._timesteps +=1

    def get_time(self):
        return self._timesteps
