#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.27
# Modified    :   2017.6.27
# Version     :   1.0



import gym
from gym import envs
from gym import wrappers
import time
envs = envs.registry.all()

# class EnvSpec(__builtin__.object)
#  |  A specification for a particular instance of the environment. Used
#  |  to register the parameters for official evaluations.
#  |  
#  |  Args:
#  |      id (str): The official environment ID
#  |      entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
#  |      trials (int): The number of trials to average reward over
#  |      reward_threshold (Optional[int]): The reward threshold before the task is considered solved
#  |      local_only: True iff the environment is to be used only on the local machine (e.g. debugging envs)
#  |      kwargs (dict): The kwargs to pass to the environment class
#  |      nondeterministic (bool): Whether this environment is non-deterministic even after seeding
#  |      tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags
#  |  
#  |  Attributes:
#  |      id (str): The official environment ID
#  |      trials (int): The number of trials run in official evaluation
#  |  
#  |  Methods defined here:
#  |  
#  |  __init__(self, id, entry_point=None, trials=100, reward_threshold=None, local_only=False, kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None, max_episode_seconds=None, timestep_limit=None)
#  |  
#  |  __repr__(self)
#  |  
#  |  make(self)
#  |      Instantiates an instance of the environment with appropriate kwargs
#  |  
#  |  ----------------------------------------------------------------------
#  |  Data descriptors defined here:
#  |  
#  |  __dict__
#  |      dictionary for instance variables (if defined)
#  |  
#  |  __weakref__
#  |      list of weak references to the object (if defined)
#  |  
#  |  timestep_limit

# for env in envs:
#     print env.id
#     try:
#         game = gym.make(env.id)
#         print(game.observation_space)
#     except Exception as e:
#         pass

# PongNoFrameskip-v4
# AirRaid-ramNoFrameskip-v0 :pass
# Breakout-ram-v4: pass
# Pong-ram-v4 : pass
game = 'MountainCarContinuous-v0'
env = gym.make(game)
# env = gym.make('Centipede-ram-v4')
env = wrappers.Monitor(env,'./record/'+game)
print("ob space is %s, action space is %s"%(env.observation_space,env.action_space.n))
    # game = gym.make(str(env))
# env.reset()

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action

# space.   
# print(env.action_space)
# #> Discrete(2)
    

# from gym import spaces
# space = spaces.Discrete(8)
# print(space)
# print(type(env.action_space.sample()))

#> Box(4,)
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):
        # env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("!!!!!!!!!!!!!!!!Episode finished after {} timesteps".format(t+1))
            print ("time is %s ac is %s ob is %s \n reward is %s \n info is %s \n"%(t,action,observation,reward,info))

            time.sleep(2)
            env.reset()
        else:
            if reward != -1:
                print ("time is %s ac is %s ob is %s \n reward is %s \n info is %s \n"%(t,action,observation,reward,info))