#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.7.23
# Modified    :   2017.7.23
# Version     :   1.0

import os
import sys
location = str(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir))) + '/'
sys.path.append(location)
import argparse

# import Main
def run_experiment(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--repeat_times', type=int, default=4,
	                   help='experiment times')
	parser.add_argument(
	    '--depth_min', type=int, default=3, help='min deap of tree')
	parser.add_argument(
	    '--depth_max', type=int, default=4, help='max deap of tree')
	parser.add_argument(
	    '--forest_min', type=int, default=1, help='min amount of tree')
	parser.add_argument(
	    '--forest_max', type=int, default=1, help='max amount of tree')

	parser.add_argument(
	    '--game_name', type=str, default='CartPole-v1',  help='game name')
	parser.add_argument(
	    '--test_circle', type=int,default=10000, help='test circle')
	parser.add_argument(
	    '--test_time_step', type=int, default=20000, help='test time step amount')
	parser.add_argument(
	    '--iter_times', type=int, default=1500000, help='test time step amount')

	args = parser.parse_args(argv[1:])
	# usage:
	# print(args)
	# output: Namespace(depth_max=5, depth_min=2, forest_max=5, forest_min=1, game_name=None, repeat_time=5, test_cicle=20000, test_time_step=100000)
	# print(args.test_cicle)
	# output:20000
	import Main
	skepj = 1
	skepi = 1
	for j in range(args.forest_min,args.forest_max+skepj, skepj):
		for i in range(args.depth_min,args.depth_max+skepi, skepi):
			for t in range(0,args.repeat_times):
				config ={
					'times':t,
					'depth':i,
					'forest_size':j,
					'game_name':args.game_name,
					'test_circle':args.test_circle,
					'test_time_step':args.test_time_step,
					'iter_times':args.iter_times
				}
				Main.main(config)


if __name__ == "__main__":
	run_experiment(sys.argv)
