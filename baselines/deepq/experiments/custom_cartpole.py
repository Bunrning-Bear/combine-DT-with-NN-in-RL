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
import time
import csv
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq import models
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.schedules import LinearSchedule
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame, EpisodicLifeEnv, ClippedRewardsWrapper

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=24, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

def parse_args():
    parser = argparse.ArgumentParser("Evaluate an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    return parser.parse_args()

def main(name_scope):
    with U.make_session(8):

        prioritized_replay = True
        prioritized_replay_alpha = 0.6
        prioritized_replay_beta0 = 0.4
        prioritized_replay_beta_iters = None
        prioritized_replay_eps = 1e-6
        buffer_size=50000
        batch_size =32
        # use cnn reduce after 1e5 time step
        model_list = [372, 64]# pong failed [256,32]# [512,128,32]# # 
        model_type = 'mlp_'+str(model_list)
        exp_type = 'baselines'
        # game = "Boxing-ram-v4" # 15w , 128->18
        game = "Boxing-ram-v4" # just soso, 128->9
        # game = "Pong-ram-v4" # just soso, 128->12
        itera_times = 2000000
        # game = "AirRaid-ramNoFrameskip-v0"
        start_exp_time = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
        exp_file_name = 'exp_%s_game_%s_model_%s[gamma=0.99][new-prioritized][simple-reward]/' % (
            exp_type, game, model_type)


        test_points = 100
        test_circle = itera_times/test_points
        # test_time_step = 50000
        GAME_NAME = game
        exploration_fraction=0.1
        train_freq = 4
        model = deepq.models.mlp(model_list)
        # model = deepq.models.cnn_to_mlp(
        #     convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        #     hiddens=[256],
        #     dueling=True
        # )
        os.makedirs(os.path.dirname(
            exp_file_name), exist_ok=True)
        record_path = exp_file_name+'test-time: %s' % start_exp_time
        scv_f = open(record_path, 'w')
        csvfile = csv.writer(scv_f)
        # Create the environment
        env = ClippedRewardsWrapper(ScaledFloatFrame(EpisodicLifeEnv(gym.make(game))))

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,# models.mlp([400,200,50]),
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=25e-5),# 5e-4
            gamma=0.99,
            scope=name_scope
        )

        '''
        act 是决策函数，用于做行为预测
        train 是训练函数，用于做参数更新
        update target 应该是更新 target网络何当前训练网络的函数:
            update_target_fn will be called periodically to copy Q network to target Q network
        我现在要做的，就是把这些函数嵌入原本写好的类中。完成11映射的关系
        '''
        # Create the replay buffer
        # replay_buffer = ReplayBuffer(50000)
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = itera_times
            beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(buffer_size)

        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=exploration_fraction*itera_times, initial_p=1.0, final_p=0.01)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        # unlimited loop
        for time_step in range(1,itera_times):
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(time_step))[0]
            # print("action select is %s"%action)
            new_obs, rew, done, _ = env.step(action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = time_step > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            # if is_solved:
            #     # Show off the result
            #     env.render()
            # else:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if time_step > 1000 and time_step % train_freq == 0:
                # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                # train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(time_step))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)


            # Update target network periodically.
            if time_step % 5000 == 0:
                print("update model... time_step %s"%time_step)
                update_target()

            if done and len(episode_rewards) % 10 == 0:
                # show table to console
                logger.record_tabular("steps", time_step)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-51:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(time_step)))
                logger.dump_tabular()

            if time_step % test_circle == 0 and len(episode_rewards) > 40:
                # test_episode_rewards =episode_rewards[30:-1]
                # record = [time_step]+test_episode_rewards
                # print("episode times %s, time_step %s, max reward %s, min reward %s, avg %s"
                #       % (len(test_episode_rewards), test_time_step, max(test_episode_rewards), min(test_episode_rewards), round(np.mean(test_episode_rewards[:]), 1)))
                # csvfile.writerow(record)
                current_ob = obs
                # (current_feature, current_state) = forest_agent.getInitState()
                test_episode_rewards = [0.0]

                another_engine = ScaledFloatFrame(EpisodicLifeEnv(gym.make(GAME_NAME)))
                test_ob = another_engine.reset()
                episode_times = 0
                while episode_times < 40:
                    test_action = act(test_ob[None], update_eps=0.01)[0]
                    test_next_ob, test_reward, test_terminal, _ = another_engine.step(test_action)
                    test_ob = test_next_ob

                    test_episode_rewards[-1] += test_reward

                    if test_terminal:
                        test_ob = another_engine.reset()
                        test_episode_rewards.append(0)
                        episode_times += 1
                        if episode_times % 10 == 0:
                            print("test episode %s"%episode_times)
                        # forest_agent.setInitState(test_ob)
                    else:
                        pass
                        # another_engine.render()
                    # if circle % print_internal == 0:
                    #     print("testing time_step %s" % circle)
                another_engine.close()
                ob = current_ob
                test_episode_rewards = test_episode_rewards[:-1]
                record = [time_step]+test_episode_rewards

                print("episode times %s, time_step %s, max reward %s, min reward %s, avg %s"
                      % (len(test_episode_rewards), time_step, max(test_episode_rewards), min(test_episode_rewards), round(np.mean(test_episode_rewards[:]), 1)))
                csvfile.writerow(record)
        scv_f.close()

if __name__ == '__main__':
    repeat = 3
    for i in range(repeat):
        main(str(i))