#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.6.29
# Version     :   1.0


import cv2
import sys
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
# sys.path.append("DQN/game/")
import gym
gym.undo_logger_setup()

from Global_Function import list_to_dic, dic_to_list
from Global_Variables import GAME_NAME, FOREST_SIZE, REWARD,ACTION,TERMINAL
from Global_Variables import REWARD_GOAL
from Base_Data_Structure import DataFeature
from Data_Sample import simple_sampling, get_features_from_origin_sample
from Agent import ForestAgent

from combine_baselines import logger
# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def main():
    # Step 1: init BrainDQN\

    # Step 2: init Game, sampling.
    game_engine = gym.make(GAME_NAME)
    sampling_amount = 100
    file_name = 'dataset/'+GAME_NAME+'-sample'+str(sampling_amount)+'csv'
    data_range = list(zip(game_engine.observation_space.low, game_engine.observation_space.high))
    data_range = list_to_dic(data_range)
    if os.path.isfile(file_name):
        sample_data = DataFeature(file_name,actions=game_engine.action_space.n,observations=game_engine.observation_space.shape[0], data_range=data_range)
    else:
        simple_sampling(game_engine, file_name, sampling_amount)
        sample_data = DataFeature(file_name,actions=game_engine.action_space.n,observations=game_engine.observation_space.shape[0], data_range=data_range)
    # build forest data structure.
    forest_agent = ForestAgent(sample_data, FOREST_SIZE)
    
    forest_agent.build()

    # initial model
    # data_iter = iter(sample_data)
    # first_origin_sample = data_iter.__next__()
    # feature = get_features_from_origin_sample(first_origin_sample)
    # observation = dic_to_list(feature)
    # forest_agent.setInitState(observation)
    # for data in data_iter:
    #     feature = get_features_from_origin_sample(data)
    #     observation = dic_to_list(feature)
    #     record={
    #         'observation': observation,
    #         'feature': feature,
    #         REWARD: data[REWARD],
    #         TERMINAL: data[TERMINAL],
    #         ACTION: data[ACTION]
    #     }
    #     forest_agent.set_replay_buffer(record)
    # initial env and agent

    observation = game_engine.reset()
    forest_agent.setInitState(observation)

    t = 0
    end_times = 0
    accumlate_amount = 0
    accumlate_time = 0.
    accumlate_time_list =[]
    max_times=0
    current_times = 0
    episode_rewards = [0.0]
    while 1!= 0:
        # game_engine.render()
        action = forest_agent.predict()
        nextObservation,reward,terminal,_ = game_engine.step(action)
        record={
            'observation': nextObservation,
            'feature': list_to_dic(nextObservation),
            REWARD: reward,
            TERMINAL: terminal,
            ACTION: action
        }
        forest_agent.set_replay_buffer(record)

        episode_rewards[-1] += reward # calculate total reward in single episode
        if terminal:
            observation = game_engine.reset()
            episode_rewards.append(0)
            forest_agent.setInitState(observation)
        is_solved = np.mean(episode_rewards[-101:-1]) >= REWARD_GOAL
        if is_solved:
            # Show off the result
            game_engine.render()
        else:
            forest_agent.update_model()
            # forest_agent.update_to_all_model()
        if terminal and len(episode_rewards) % 10 == 0:
            # show table to console
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.dump_tabular()
        t = t + 1
    #     # nextObservation = preprocess(nextObservation)
        
    #     # brain.setPerception(nextObservation,action,reward,terminal)
        
    #     # if reward != 1:
    #     #     time.sleep(0.5)
    #     # print "action is %s, reward is %s"%(action,reward)


    #     time_count +=1
    #     current_times+=1
    #     if terminal:
    #         end_times +=1
    #         max_times = max(max_times,current_times)
    #         current_times = 0
    #         if end_times % 100 == 0:
    #             accumlate_amount = accumlate_amount + 1.
    #             ave = (time_count)/100.0
    #             accumlate_time = accumlate_time * (accumlate_amount - 1.) / accumlate_amount + ave / accumlate_amount
    #             print "[end game] time_count : %s,accumlate_time :%s accumlate_amount %s max times %s"%(ave,accumlate_time, accumlate_amount,max_times)
    #             max_times  = 0
    #             time_count = 0
    #         if end_times % 2000 == 0:
    #             accumlate_time_list.append(accumlate_time)
    #             print "accumlate_time_list %s"%accumlate_time_list[-10:]
    #             accumlate_time_list = accumlate_time_list[-10:]
    #             accumlate_time = 0
    #             accumlate_amount = 0
    #         # time.sleep(3)
    #         observation = game_engine.reset()
    #         forest_agent.setInitState(observation)

    # # forest_agent = initial_model()

    # # Step 3: play game
    # # Step 3.1: obtain init state
    # action0 = np.array([1,0])  # do nothing
    # observation0, reward0, terminal = ganme_engine.frame_step(action0)
    # observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    # # initial action amount and observation amount
    # forest_agent.setInitState(observation0)

    # # Step 3.2: run the game
    # while 1!= 0:
    #     action = forest_agent.predict()
    #     nextObservation,reward,terminal = ganme_engine.frame_step(action)
    #     nextObservation = preprocess(nextObservation)
    #     record = {
    #         'action':action,
    #         'reward':reward,
    #         'terminal':terminal,
    #         'next_observation':nextObservation
    #     }
    #     forest_agent.set_replay_buffer(record)
    #     forest_agent.update_model()

if __name__ == '__main__':
    main()