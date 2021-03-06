#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.7.6
# Modified    :   2017.7.6
# Version     :   1.0


import os
import sys
location = str(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(
            os.path.dirname(__file__), os.pardir),
            os.pardir),os.pardir))) + '/'
sys.path.append(location)


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

# def model(inpt, num_actions, scope, reuse=False):
#     """This model takes as input an observation and returns values of all actions."""
#     with tf.variable_scope(scope, reuse=reuse):
#         out = inpt
#         out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
#         out = layers.fully_connected(out, num_outputs=24, activation_fn=tf.nn.tanh)
#         out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
#         return out


if __name__ == '__main__':
    game_name = "CartPole-v0"
    agent_name = 'agent_test'
    agent_name2 = 'agent_test_2'
    session = U.MultiSession(agent_name)
    session2 = U.MultiSession(agent_name2)
    path = 'saved_networks/'+game_name+'-'+agent_name
    path2='saved_networks/'+game_name+'-'+agent_name2
    # Create the environment
    env = gym.make(game_name)
    # with U.make_session(8):
    g1 = tf.Graph()
    with tf.variable_scope(agent_name):
        with g1.as_default():

            session.make_session(g1,4)
            # Create all the functions necessary to train the model
            act, train, update_target, debug = deepq.build_train(
                session=session,
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            )
            t_val = tf.Variable(0)    
            session.initialize()
            session.init_saver()
            
    with tf.variable_scope(agent_name2):
        g2 = tf.Graph()
        with g2.as_default():
            session2.make_session(g2,4)
            # Create all the functions necessary to train the model
            act2, train2, update_target2, debug = deepq.build_train(
                session=session2,
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            )
            t_val2 = tf.Variable(0)
            session2.initialize()
            session2.init_saver()
        '''
        act 是决策函数，用于做行为预测
        train 是训练函数，用于做参数更新
        update target 应该是更新 target网络何当前训练网络的函数:
            update_target_fn will be called periodically to copy Q network to target Q network
        我现在要做的，就是把这些函数嵌入原本写好的类中。完成11映射的关系
        '''
    # # Create the replay buffer
    replay_buffer = ReplayBuffer(50000)
    replay_buffer2 = ReplayBuffer(50000)
    # # Create the schedule for exploration starting from 1 (every action is random) down to
    # # 0.02 (98% of actions are selected according to values predicted by the model).
    exploration = LinearSchedule(schedule_timesteps=20000, initial_p=1.0, final_p=0.02)
    exploration2 = LinearSchedule(schedule_timesteps=20000, initial_p=1.0, final_p=0.02)


    # # Initialize the parameters and copy them to the target network.



    # print(session.sess.run(t_val))
    # session.save_state(path, 0)
    update_target()
    update_target2()
    session.load_state(path)
    session2.load_state(path2)

    t = session.sess.run(t_val)
    t2 = session2.sess.run(t_val2)
    assert t == t2, 't!=t2, impossible!'
    print("initial t=%s"%t)
    episode_rewards = [0.0]
    episode_rewards_2 = [0.0]
    obs = env.reset()
    # unlimited loop
    action_different = 0
    while(True):
        t = t + 1
        # Take action and update exploration to the newest value
        action = act(obs[None], update_eps=exploration.value(t))[0]
        action2 = act(obs[None], update_eps=exploration.value(t))[0]
        if action != action2:
            action_different+=1
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done),obs)
        replay_buffer2.add(obs, action, rew, new_obs, float(done),obs)
        obs = new_obs
        episode_rewards[-1] += rew # calculate total reward in single episode
        episode_rewards_2[-1] += rew # calculate total reward in single episode
        if done:
            print("action different times %s "%action_different)
            action_different = 0
            obs = env.reset()
            episode_rewards.append(0)
            episode_rewards_2.append(0)

        is_solved = np.mean(episode_rewards[-101:-1]) >= 200
        
        if is_solved:
            print("solved")
            # Show off the result
            env.render()
        else:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 1000:
                data_list = replay_buffer.sample(64)
                # distribute

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
                train2(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
            # Update target network periodically.
            if t % 1000 == 0:
                print("update target")
                update_target()
            if t % 5000 == 0:
                print("saves")
                session.sess.run(tf.assign(t_val,t))
                session.save_state(path, t)                
                session2.sess.run(tf.assign(t_val2,t))
                session2.save_state(path2, t)      

        if done and len(episode_rewards) % 10 == 0:
            # show table to console
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", len(episode_rewards))
            logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
            logger.record_tabular("% time spent exploring", int(100.0 * exploration.value(t)))
            logger.dump_tabular()
