#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.30
# Modified    :   2017.6.30
# Version     :   1.0

from Global_Variables import REWARD, ACTION, TERMINAL, STATE_ATTRS

def simple_sampling(env,file_name,sampling_amount):
    observations = env.observation_space
    head_line = _get_head_line(observations.shape[0])
    w = open(file_name,'w')
    head_line += '\n'
    w.write(head_line)
    observation = env.reset()
    env.render()
    for i in range(0,sampling_amount):
        env.render()
        action = env.action_space.sample()
        nextObservation,reward,terminal,info = env.step(action)
        ob_list = list(nextObservation)
        ob_list.append(float(reward))
        ob_list.append(int(action))
        ob_list.append(int(terminal))
        str_list = [str(i) for i in ob_list]
        print str_list
        content = ",".join(str_list)# join(line_list[0:10]+[line_list[-1]])
        content +='\n'
        w.write(content)
        if terminal:
            observation = env.reset()
            

def _get_head_line(ob_size):
    content = ""
    for i in range(0,ob_size):
        content+=str(i)+':continuous,'

    content +=REWARD+':continuous,'+ACTION+':nominal:class,'+TERMINAL+':discrete'
    return content

def get_features_from_origin_sample(sample):
    feature = sample.copy()
    for key,value in feature.items():
        if key in STATE_ATTRS:
            del feature[key]
    return feature
