#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.30
# Modified    :   2017.6.30
# Version     :   1.0


# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
# sys.path.append("game/")
# import wrapped_flappy_bird as game
# from BrainDQN_origin import BrainDQN
import gym
from BrainDQN_Nature import BrainDQN
import time 
import numpy as np

# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def playFlappyBird():
    # Step 1: init BrainDQN
    actions = 2
    # brain = BrainDQN(actions)
    # Step 2: init Flappy Bird Game
    env = gym.make('CartPole-v0')
    actions = env.action_space
    observations = env.observation_space

    # flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    # action0 = np.array([1,0])  # do nothing
    # observation0, reward0, terminal = flappyBird.frame_step(action0)
    # observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain = BrainDQN(actions.n, observations.shape[0], BrainDQN.SIMPLE_OB,agent_name='CartPole-v0')

    observation = env.reset()
    brain.setInitState(observation)
    env.render()
    # action = env.action_space.sample()
    # observation, reward, done, info = env.step(action)


    # Step 3.2: run the game
    time_count = 0
    end_times = 0
    accumlate_amount = 0
    accumlate_time = 0.
    accumlate_time_list =[]
    max_times=0
    current_times = 0
    while 1!= 0:
        env.render()
        action = brain.getAction()
        nextObservation,reward,terminal,info = env.step(int(max(action)))
        # nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,terminal)
        
        # if reward != 1:
        #     time.sleep(0.5)
        # print "action is %s, reward is %s"%(action,reward)


        time_count +=1
        current_times+=1
        if terminal:
            end_times +=1

            max_times = max(max_times,current_times)
            current_times = 0
            if end_times % 100 == 0:
                accumlate_amount = accumlate_amount + 1.
                ave = (time_count)/100.0
                accumlate_time = accumlate_time * (accumlate_amount - 1.) / accumlate_amount + ave / accumlate_amount
                print "[end game] time_count : %s,accumlate_time :%s accumlate_amount %s max times %s"%(ave,accumlate_time, accumlate_amount,max_times)
                max_times  = 0
                time_count = 0
            if end_times % 2000 == 0:
                accumlate_time_list.append(accumlate_time)
                print "accumlate_time_list %s"%accumlate_time_list[-10:]
                accumlate_time_list = accumlate_time_list[-10:]
                accumlate_time = 0
                accumlate_amount = 0

            
            # time.sleep(3)
            observation = env.reset()
            brain.setInitState(observation)


def main():
    playFlappyBird()

if __name__ == '__main__':
    main()