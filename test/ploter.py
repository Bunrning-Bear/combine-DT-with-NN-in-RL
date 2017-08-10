#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.7.23
# Modified    :   2017.7.23
# Version     :   1.0
import csv
import numpy as np
import matplotlib.pyplot as plt 
import os

def cal_avg_reward(row):
    time = int(row[0])
    num_row = [float(item) for item in row[1:]]
    avg_reward = sum(num_row)/float(len(num_row))
    return (time, avg_reward)


def get_x_and_y(csvfile):
    x = []
    y = []
    for row in csvfile:
        time, avg_reward = cal_avg_reward(row)
        x.append(time)
        y.append(avg_reward)
    return (x, y)


def plot_single_exp(complete_exp_name,color):
    y_list = []
    x = []

    files = os.listdir(complete_exp_name)
    for file in files:
        print("in file %s"%file)
        complete_file_name = complete_exp_name+file
        if os.path.getsize(complete_file_name) == 0:
            print("empty file")
            continue
        with open(complete_file_name) as f:
            reader = csv.reader(f)
            x_temp, y = get_x_and_y(reader)
            if x == []:
                x = x_temp
            if x != [] and len(x) > len(x_temp):
                x = x_temp
            y_list.append(np.array(y))
    print("x min is %s"%len(x))
    y_mat =[]
    for i in range(0,len(y_list)):
        y_mat.append(np.array(y_list[i][-len(x):]))
    y_mat = np.array(y_mat)
    print(y_mat)
    # for i in range(0,len(x)):
    #         print("time is %s"item)
    # print(y_mat)
    # y_avg = np.mean(y_mat)
    y_max = list(y_mat.max(0))
    y_avg = np.mean(y_mat,axis=0)
    y_min = list(y_mat.min(0))
    plt.plot(x,y_avg,color+'x-')
    plt.plot(x,y_max,color+'x--')
    plt.plot(x,y_min,color+'x--')
    plt.grid(True)
    # plt.show()

def create_dir(exp,iter_times,game,mlp_type,postfix,depth=0,forest=1):
    print("depth type %d"%(depth))
    print("forest type %d"%(forest))
    print("iter_times type %d"%(iter_times))
    
    if exp == 'dt':
        directory = './record/exp_dt_iter_%d_game_%s_model_mlp_%s_depth_%d_forest_%d%s/'%(iter_times,game,mlp_type,depth,forest,postfix)
    else:
        directory = '/home/burningbear/Projects/machine_learning/conbine-DT-with-NN-in-RL/baselines/deepq/experiments/exp_baselines_game_%s_model_mlp_%s%s/'%(game,mlp_type,postfix)
    print("select dir %s"%directory)
    return directory
def main():
    import os
    # game boxing
    # exp = 'exp_dt_iter_1000000_game_Boxing-ram-v4_model_mlp_[372.64]_depth_1_forest_1'
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372, 64]','[prioritized]',1,1)
    # plot_single_exp(directory,'b')
    # exp = 'exp_dt_iter_1000000_game_Boxing-ram-v4_model_mlp_[372.64]_depth_0_forest_1[prioritized]'
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372.64]','[prioritized]',0,1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372, 64]','[prioritized]',2,1)
    # plot_single_exp(directory,'m')
    directory = create_dir('bs',1000000,'Boxing-ram-v4','[372, 64]','[gamma=0.99][prioritized][simple-reward]',0,1)
    plot_single_exp(directory,'b')
    directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][update-all]',2,1)
    plot_single_exp(directory,'r')
    directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][update-all]',3,1)
    plot_single_exp(directory,'m')
    # game crazyClimber
    # directory = create_dir('bs',3000000,'CrazyClimber-ram-v4','[376, 64]','[gamma=0.99][prioritized]')
    # plot_single_exp(directory,'b')
    # directory = create_dir('bs',3000000,'CrazyClimber-ram-v4','[376, 50]','[gamma=0.99][prioritized][simple-reward]')
    # plot_single_exp(directory,'g')

    # game pong
    # directory = create_dir('bs',3000000,'Pong-ram-v4','[376, 64]','[gamma=0.99][prioritized]')
    # plot_single_exp(directory,'b')
    # directory = create_dir('bs',3000000,'Pong-ram-v4','[376, 50]','[gamma=0.99][prioritized][simple-reward]')
    # plot_single_exp(directory,'g')

    # game airRaid
    # directory = create_dir('bs',3000000,'AirRaid-ram-v4','[376, 64]','[gamma=0.99][prioritized]')
    # plot_single_exp(directory,'b')
    # directory = create_dir('bs',3000000,'AirRaid-ram-v4','[376, 64]','[gamma=0.99][prioritized][simple-reward]')
    # plot_single_exp(directory,'g')

    plt.show()
    
if __name__ == '__main__':
    main()