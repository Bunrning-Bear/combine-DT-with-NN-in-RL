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


def plot_single_exp(complete_exp_name, color):
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
        y_mat.append(np.array(y_list[i][:len(x)]))
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
    # plt.plot(x,y_max,color+'x--')
    # plt.plot(x,y_min,color+'x--')
   
    # plt.show()

def plot_every_exp(complete_exp_name, color_list):
    files = os.listdir(complete_exp_name)
    for index, file in enumerate(files):
        print("in file %s"%file)
        complete_file_name = complete_exp_name+file
        if os.path.getsize(complete_file_name) == 0:
            print("empty file")
            continue
        with open(complete_file_name) as f:
            reader = csv.reader(f)
            x_temp, y = get_x_and_y(reader)
            plt.plot(x_temp,y,color_list[index % len(color_list)-1]+'x--')

def create_dir(exp,iter_times,game,mlp_type,postfix,depth=0,forest=1, splite_amount=None, need_iter=True):
    print("depth type %d"%(depth))
    print("forest type %d"%(forest))
    print("iter_times type %d"%(iter_times))
    
    if exp == 'dt':
        directory = './record/exp_dt_iter_%d_game_%s_model_mlp_%s_depth_%d_forest_%d%s/'%(iter_times,game,mlp_type,depth,forest,postfix)
        if splite_amount == None:
            directory = './record/exp_dt_iter_%d_game_%s_model_mlp_%s_depth_%d_forest_%d%s/'%(iter_times,game,mlp_type,depth,forest,postfix)
        else:
            directory = './record/exp_dt_iter_%d_game_%s_model_mlp_%s_depth_%d_forest_%d_split_%s%s/'%(iter_times,game,mlp_type,depth,forest,splite_amount,postfix)

    else:
        prefix = '/home/burningbear/Projects/machine_learning/conbine-DT-with-NN-in-RL/baselines/deepq/experiments/'
        if splite_amount == None:
            if need_iter:
                directory = prefix+'exp_baselines_game_%s_model_mlp_%s_iter_%s%s/'%(game,mlp_type, iter_times,postfix)
            else:
                directory = prefix+'exp_baselines_game_%s_model_mlp_%s%s/'%(game,mlp_type,postfix)
        else:
            directory = prefix+'exp_baselines_game_%s_model_mlp_%s_iter_%s_split_%s%s/'%(game,mlp_type, iter_times,splite_amount, postfix)
    print("select dir %s"%directory)
    return directory
def main():
    import os
    color_list = ['k','g','y','r','b']

    # game boxing
    # exp = 'exp_dt_iter_1000000_game_Boxing-ram-v4_model_mlp_[372.64]_depth_1_forest_1'
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372, 64]','[prioritized]',1,1)
    # plot_single_exp(directory,'b')
    # exp = 'exp_dt_iter_1000000_game_Boxing-ram-v4_model_mlp_[372.64]_depth_0_forest_1[prioritized]'
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372.64]','[prioritized]',0,1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1000000,'Boxing-ram-v4','[372, 64]','[prioritized]',2,1)
    # plot_single_exp(directory,'m')
    # # baseline
    # directory = create_dir('bs',1000000,'Boxing-ram-v4','[372, 64]','[gamma=0.99][prioritized][simple-reward]',0,1,need_iter=False)
    # plot_single_exp(directory,'b')
    # baseline with not priority repley buffer
    # directory = create_dir('bs',2000000,'Boxing-ram-v4','[372, 64]','[gamma=0.99][real-no-prioritized][simple-reward]',0,1,need_iter=False)
    # plot_single_exp(directory,'k')
    # directory = create_dir('bs',2000000,'Boxing-ram-v4','[372, 64]','[gamma=0.99][new-prioritized][simple-reward]',3,1)
    # plot_single_exp(directory,'k')

    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][initial][update-all]',3,1)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][update-all]',3,1)
    # plot_single_exp(directory,'m')
    # directory = create_dir('dt',1500000,'Boxing-ram-v4','[372, 64]','[prioritized][initial][update-all]',4,1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][initial][update-all]',0,1)
    # plot_single_exp(directory,'y')
    
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][not-initial-clear][update-all]',0,1)
    # plot_single_exp(directory,'y')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[prioritized][initial-clear][update-all]',0,1)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][prioritized][not-initial-clear][update-all]',3,1)
    # plot_single_exp(directory,'g')
    # 
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][prioritized][not-initial-clear][update-all][failed-prior-duplicate-update]',3,1)
    # plot_single_exp(directory,'k')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][single-prioritized][not-initial-clear]',2,1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][single-prioritized][not-initial-clear][update-all]',2,1)
    # plot_single_exp(directory,'y')
    # # # 
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][single-prioritized][not-initial-clear][update-all][simple]',2,1)
    # plot_single_exp(directory,'m')
    # # # 
    # directory = create_dir('dt',2000000,'Boxing-ram-v4','[372, 64]','[global-time][random-prioritized][not-initial-clear][update-all]',2,1)
    # plot_single_exp(directory,'r')
    
    # [global-time][single-prioritized][not-initial-clear][update-all][simple]
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

    # game cartpole
    # directory = create_dir('bs',1000000,'CartPole-v1','[128, 32]','[gamma=0.99][no-prioritized][simple-reward]')
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'b')
    ## update to all node, success
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear][update-all]',4,1)
    # plot_single_exp(directory,'r')
    # plot_every_exp(directory,color_list)

    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear][update-all]',3,1)
    # plot_single_exp(directory,'k')
    # plot_every_exp(directory,color_list)
    ## update to all node, success


    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear]',0,1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear]',1,1,splite_amount=100)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][initial-clear]',1,1,splite_amount=10)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,1,splite_amount=10)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][initial-clear]',1,1,splite_amount=10)
    # plot_single_exp(directory,'b')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear]',2,1)
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'g')


    # forest exp
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,1,splite_amount=1)
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'r')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,2,splite_amount=1)
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,3,splite_amount=1)
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'y')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,4,splite_amount=1)
    # plot_single_exp(directory,'k')
    # plot_every_exp(directory,color_list)

    # splite exp
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,1,splite_amount=10)
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear]',1,1,splite_amount=15)
    # plot_single_exp(directory,'y')
    # plot_every_exp(directory,color_list)
    # directory = create_dir('dt',1000000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][not-initial-clear]',1,1,splite_amount=100)
    # plot_single_exp(directory,'k')  
    # plot_every_exp(directory,color_list)
    
    # depth exp
    # directory = create_dir('dt',1500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1][stable-explore]',1,1,splite_amount=1)
    # plot_single_exp(directory,'r')
    directory = create_dir('dt',2500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1]',2,1,splite_amount=1)
    plot_every_exp(directory, color_list)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',1500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1][stable-explore]',3,1,splite_amount=1)
    # plot_single_exp(directory,'y')
    # directory = create_dir('dt',1500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1][stable-explore]',4,1,splite_amount=1)
    # plot_single_exp(directory,'k')
    # directory = create_dir('dt',1500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1]',1,1,splite_amount=1)
    # plot_every_exp(directory,color_list)
    

    # single parameter exp
    # directory = create_dir('dt',2500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][update-all][split-v0.1]',0,1,splite_amount=1)
    # plot_single_exp(directory,'b')
    # directory = create_dir('dt',2500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][split-v0.1]',2,1,splite_amount=1)
    # plot_single_exp(directory,'g')
    # directory = create_dir('dt',2500000,'CartPole-v1','[128, 32]','[global-time][single-prioritized][no-initial-clear][update-all][split-v0.1]',2,1,splite_amount=1)
    # plot_single_exp(directory,'r')
    # plot_every_exp(directory,color_list)


    # game = 'Acrobot-v1'
    # directory = create_dir('bs',1000000,game,'[128, 32]','[gamma=0.99][no-prioritized][simple-reward]')
    # plot_every_exp(directory,color_list)
    # plot_single_exp(directory,'b')
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    main()