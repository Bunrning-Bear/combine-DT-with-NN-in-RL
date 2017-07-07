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

from baselines import deepq


def main():
    env = gym.make("CartPole-v0")
    act = deepq.load("cartpole_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
