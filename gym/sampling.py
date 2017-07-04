#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.30
# Modified    :   2017.6.30
# Version     :   1.0

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
# sys.path.append("game/")
# import wrapped_flappy_bird as game
# from BrainDQN_origin import BrainDQN
import gym
import numpy as np
import csv

# preprocess raw image to 80*80 gray image

def get_head_line(ob_size):
    content = ""
    for i in range(0,ob_size):
        content+=str(i)+':continuous,'
    content +='reward:continuous,action:continuous:class,terminal:discrete'
    return content

def simple_sampling(env,file_name,sampling_amount):
    observations = env.observation_space
    head_line = get_head_line(observations.shape[0])
    w = open(file_name,'w')
    w.write(head_line)
    observation = env.reset()
    env.render()
    for i in range(0,sampling_amount):
        env.render()
        action = env.action_space.sample()
        nextObservation,reward,terminal,info = env.step(action)
        ob_list = list(nextObservation)
        ob_list.append(float(reward))
        ob_list.append(float(action))
        ob_list.append(int(terminal))
        str_list = [str(i) for i in ob_list]
        print str_list
        content = ",".join(str_list)# join(line_list[0:10]+[line_list[-1]])
        content +='\n'
        w.write(content)
        if terminal:
            observation = env.reset()


def main():
    # Step 1: init BrainDQN
    actions = 2
    # brain = BrainDQN(actions)
    # Step 2: init Flappy Bird Game

    game_name = 'CartPole-v0'
    env = gym.make(game_name)
    simple_sampling(env, game_name+'-sample.csv', 10000)
    # flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    # action0 = np.array([1,0])  # do nothing
    # observation0, reward0, terminal = flappyBird.frame_step(action0)
    # observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    # brain = BrainDQN(actions.n, observations.shape[0], BrainDQN.SIMPLE_OB,agent_name=game_name)

    # action = env.action_space.sample()
    # observation, reward, done, info = env.step(action)


    # Step 3.2: run the game
    # time_count = 0
    # end_times = 0
    # accumlate_amount = 0
    # accumlate_time = 0.
    # while 1!= 0:
    #     env.render()
    #     action = env.action_space.sample()
    #     nextObservation,reward,terminal,info = env.step(action)
    #     # nextObservation = preprocess(nextObservation)
    #     brain.setPerception(nextObservation,action,reward,terminal)
        
        # if reward != 1:
        #     time.sleep(0.5)
        # print "action is %s, reward is %s"%(action,reward)


        # time_count +=1
        # if terminal:
        #     end_times +=1
        #     if end_times % 100 == 0:
        #         accumlate_amount = accumlate_amount + 1.
        #         ave = (time_count)/100.0
        #         accumlate_time = accumlate_time * (accumlate_amount - 1.) / accumlate_amount + ave / accumlate_amount
        #         print "[end game] time_count : %s,accumlate_time :%s accumlate_amount %s"%(ave,accumlate_time, accumlate_amount)
        #         time_count = 0
            
        #     # time.sleep(3)
        #     observation = env.reset()
        #     brain.setInitState(observation)


if __name__ == '__main__':
    main()
