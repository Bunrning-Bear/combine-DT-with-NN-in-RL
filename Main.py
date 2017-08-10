#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.7.23
# Version     :   1.0

import os
import logging
import numpy as np
import csv
import time
logging.basicConfig(level=logging.INFO)
# sys.path.append("DQN/game/")
import gym
gym.undo_logger_setup()

from Global_Function import list_to_dic, dic_to_list
from Global_Variables import REWARD, ACTION, TERMINAL
from Global_Variables import REWARD_GOAL, OBSERVE
from Global_Variables import RECORD_PREFIX_PATH
from Base_Data_Structure import DataFeature
from Data_Sample import simple_sampling, get_features_from_origin_sample
from Agent import ForestAgent
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame, EpisodicLifeEnv, ClippedRewardsWrapper

from combine_baselines import logger
# preprocess raw image to 80*80 gray image
# def preprocess(observation):
#     observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
#     ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
#     return np.reshape(observation,(80,80,1))


def main(config):
    # initial constant value
    MODEL_LIST = [372, 64]
    model_type = 'mlp_'+str(MODEL_LIST)
    exp_type = 'dt'
    use_gpu = 'gpu'
    start_exp_time = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
    exp_file_name = 'exp_%s_iter_%s_game_%s_model_%s_depth_%s_forest_%s_%s[prioritized][initial][update-all]/' % (
        exp_type, config['iter_times'], config['game_name'], model_type, config['depth'], config['forest_size'],use_gpu)
    test_points = 100
    test_circle = config['iter_times']/test_points
    GAME_NAME = config['game_name']
    FOREST_SIZE = config['forest_size']
    train_freq = 4

    #set record file
    os.makedirs(os.path.dirname(
        RECORD_PREFIX_PATH+exp_file_name), exist_ok=True)
    record_path = RECORD_PREFIX_PATH+exp_file_name+'test-time: %s' % start_exp_time
    scv_f = open(record_path, 'w')
    csvfile = csv.writer(scv_f)

    # initial game agent
    game_engine = ClippedRewardsWrapper(ScaledFloatFrame(EpisodicLifeEnv(gym.make(GAME_NAME))))

    # get initial sample data.
    sampling_amount = 15
    file_name = '../dataset/'+GAME_NAME+'-sample'+str(sampling_amount)+'prioritized'+'.csv'
    # TODO for [0,255] only
    data_range = list(zip(game_engine.observation_space.low/255,
                          game_engine.observation_space.high/255))
    data_range = list_to_dic(data_range)
    if os.path.isfile(file_name):
        sample_data = DataFeature(file_name, actions=game_engine.action_space.n,
                                  observations=game_engine.observation_space.shape[0], data_range=data_range)
    else:
        simple_sampling(game_engine, file_name, sampling_amount)
        sample_data = DataFeature(file_name, actions=game_engine.action_space.n,
                                  observations=game_engine.observation_space.shape[0], data_range=data_range)

    # build forest data structure.
    forest_agent = ForestAgent(
        sample_data, config, exp_file_name+'test-time: %s/' % start_exp_time,itera_times=config['iter_times'], model_type=MODEL_LIST, use_gpu=use_gpu)
    logging.info("before build ")
    forest_agent.build()

    # initial model
    data_iter = iter(sample_data)
    first_origin_sample = data_iter.__next__()
    feature = get_features_from_origin_sample(first_origin_sample)
    observation = dic_to_list(feature)
    forest_agent.setInitState(observation)
    for data in data_iter:
        feature = get_features_from_origin_sample(data)
        observation = dic_to_list(feature)
        record={
            'observation': observation,
            'target_ob': feature,
            REWARD: data[REWARD],
            TERMINAL: data[TERMINAL],
            ACTION: data[ACTION]
        }
        forest_agent.set_replay_buffer(record)
    # initial env and agent

    observation = game_engine.reset()
    forest_agent.setInitState(observation)
    forest_agent.initial_model()
    model_path = forest_agent.model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path+'time_log'):
        # load time steps
        f1 = open(model_path+'time_log', 'r')
        time_step = int(f1.read())
        print("loading time step %s" % time_step)
        f1.close()
        f1 = open(model_path+'time_log', 'w')
    else:
        # new time step
        print("new test")
        time_step = 0
        with open(model_path+'time_log', 'w') as f1:
            f1.write('0')

    end_times = 0
    accumlate_amount = 0
    accumlate_time = 0.
    accumlate_time_list = []
    max_times = 0
    current_times = 0

    episode_rewards = [0.0]
    print("before training")
    for iter_times in range(time_step+1, config['iter_times']):
        # game_engine.render()
        action = forest_agent.predict()
        nextObservation, reward, terminal, _ = game_engine.step(action)
        record = {
            'observation': nextObservation,
            'target_ob': list_to_dic(nextObservation),
            REWARD: reward,
            TERMINAL: terminal,
            ACTION: action
        }

        forest_agent.set_replay_buffer(record)

        # calculate total reward in single episode
        episode_rewards[-1] += reward
        if terminal:
            observation = game_engine.reset()
            episode_rewards.append(0)
            forest_agent.setInitState(observation)

        is_solved = np.mean(episode_rewards[-101:-1]) >= REWARD_GOAL
        # if is_solved:
        #     # Show off the result
        #     break
        #     game_engine.render()
        # else:
        if time_step > OBSERVE and time_step % train_freq == 0:
            # forest_agent.update_model()
            forest_agent.update_to_all_model()

        if len(episode_rewards) % 10 == 0 and terminal:
        # if time_step % 200 == 0:
            # show table to console
            logger.record_tabular("steps", time_step)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(
                np.mean(episode_rewards[-51:-1]), 1))
            logger.dump_tabular()
        time_step = time_step + 1
        # store time step
        if time_step % 5000 == 0 :
            with open(model_path+'time_log', 'w') as f1:
                print("store time step: %s"%time_step)
                f1.write(str(time_step))
            # f1.write(str(time_step))

        if time_step % test_circle == 0 and len(episode_rewards) >= 30:
            # test_episode_rewards = episode_rewards[-20:-1]
            # record = [time_step] + test_episode_rewards
            # print("episode times %s, time_step %s, max reward %s, min reward %s, avg %s"
            #       % (len(test_episode_rewards), test_time_step, max(test_episode_rewards), min(test_episode_rewards), round(np.mean(episode_rewards[:]), 1)))
            # csvfile.writerow(record)
            (current_feature, current_state) = forest_agent.getInitState()
            test_episode_rewards = [0.0]

            another_engine = ScaledFloatFrame(EpisodicLifeEnv(gym.make(GAME_NAME)))
            test_ob = another_engine.reset()
            forest_agent.setInitState(test_ob)
            episode_times = 0
            # another_engine.render()
            while episode_times < 40:
                test_action = forest_agent.predict(for_test=True)
                test_next_ob, test_reward, test_terminal, _ = another_engine.step(
                    test_action)
                test_record = {
                    'observation': test_next_ob,
                    'target_ob': list_to_dic(test_next_ob),
                    REWARD: test_reward,
                    TERMINAL: test_terminal,
                    ACTION: test_action
                }
                forest_agent.update_state(test_record)
                test_episode_rewards[-1] += test_reward

                if test_terminal:
                    test_ob = another_engine.reset()
                    test_episode_rewards.append(0)
                    forest_agent.setInitState(test_ob)
                    episode_times += 1
                    if episode_times % 10 == 0:
                        print("test episode %s"%episode_times)
                else:
                    pass
                    # another_engine.render()
            another_engine.close()
            forest_agent.restore_init_state(current_feature, current_state)
            test_episode_rewards = test_episode_rewards[:-1]
            record = [time_step]+test_episode_rewards

            print("episode times %s, time_step %s, max reward %s, min reward %s, avg %s"
                  % (len(test_episode_rewards),time_step, max(test_episode_rewards), min(test_episode_rewards), round(np.mean(test_episode_rewards[:]), 1)))
            csvfile.writerow(record)
    scv_f.close()
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

    #

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

# if __name__ == '__main__':
    # main(args)
