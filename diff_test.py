#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.7.7
# Modified    :   2017.7.7
# Version     :   1.0


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
from Base_Data_Structure import DataFeature
from Data_Sample import simple_sampling, get_features_from_origin_sample
from Agent import ForestAgent



import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import combine_baselines.common.tf_util as U
from combine_baselines import logger
from combine_baselines import deepq
from combine_baselines.deepq.replay_buffer import ReplayBuffer
from combine_baselines.common.schedules import LinearSchedule
from combine_baselines.deepq.models import model


def main():
    # Step 1: init BrainDQN\

    # Step 2: init Game, sampling.
    game_engine = gym.make(GAME_NAME)

    # ---------------------
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
    # ----------------------------


    # initial origin
    agent_name='agent_test'
    session = U.MultiSession(agent_name)
    path = 'saved_networks/'+GAME_NAME+'-'+agent_name

    # with U.make_session(8):
    g = tf.Graph()
    with tf.variable_scope(agent_name):
        with g.as_default():
            session.make_session(g,4)

            # a = tf.Variable(2)
            # session.initialize()
            # session.init_saver()
            # print(session.sess.run(a))
            # # session.save_state(path, 0)
            # session.load_state(agent_name)
            # print(session.sess.run(a))
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                session=session,
                make_obs_ph=lambda name: U.BatchInput(game_engine.observation_space.shape, name=name),
                q_func=model,
                num_actions=game_engine.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            )
            replay_buffer = ReplayBuffer(50000)
            # Create the schedule for exploration starting from 1 (every action is random) down to
            # 0.02 (98% of actions are selected according to values predicted by the model).
            exploration = LinearSchedule(schedule_timesteps=200000, initial_p=1.0, final_p=0.02)

            t_val = tf.Variable(0)    

            # # Initialize the parameters and copy them to the target network.
            session.initialize()
            session.init_saver()
            # print(session.sess.run(t_val))
            # session.save_state(path, 0)
            update_target()
            session.load_state(path)
            t = session.sess.run(t_val)


    observation = game_engine.reset()
    episode_rewards = [0.0]
    # game_engine.render()

    forest_agent.setInitState(observation)

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



    t = 0
    end_times = 0
    accumlate_amount = 0
    accumlate_time = 0.
    accumlate_time_list =[]
    max_times=0
    current_times = 0

    action_different = 0
    while(True):
        t = t + 1
        # game_engine.render()
        action = forest_agent.predict()


        action2 = act(observation[None], update_eps=exploration.value(t))[0]
        if action != action2:
            action_different+=1

        nextObservation, reward, terminal,_ = game_engine.step(action)
        replay_buffer.add(observation, action, reward, nextObservation, float(terminal),observation)
        observation = nextObservation
        

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
            print("action different times %s "%action_different)
            action_different = 0
            observation = game_engine.reset()
            episode_rewards.append(0)


            forest_agent.setInitState(observation)

        is_solved = np.mean(episode_rewards[-101:-1]) >= 200
        if is_solved:

            # Show off the result
            print("solved")
            game_engine.render()
        else:

            forest_agent.update_to_all_model()

            if t > 1000:
                data_list = replay_buffer.sample(64)
                obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
                for data in data_list:
                    # sample distribute completed
                    obs_t, action, reward, obs_tp1, done, features = data
                    obses_t.append(np.array(obs_t, copy=False))
                    actions.append(np.array(action, copy=False))
                    rewards.append(reward)
                    obses_tp1.append(np.array(obs_tp1, copy=False))
                    dones.append(done)
                # total distribute completed, before train.
                np.array(obses_t)
                np.array(actions)
                np.array(rewards)
                np.array(obses_tp1)
                np.array(dones)
                train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            if t % 1000 == 0:
                print("update target")
                update_target()
            if t % 5000 == 0:
                print("saves")
                session.sess.run(tf.assign(t_val,t))
                session.save_state(path, t)       

        if terminal and len(episode_rewards) % 10 == 0:
            # show table to console
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.dump_tabular()


if __name__ == '__main__':
    main()