# File: policy_gradient.py 
# Author(s): Rishikesh Vaishnav
# Created: 29/06/2018

import logging
import os

import gym

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

env = gym.make('CartPole-v0')
agent = RandomAgent(env.action_space)

episode_count = 100
max_steps = 200
reward = 0
done = False

while(True):
    ob = env.reset()

    for j in range(max_steps):
        env.render();
        action = agent.act(ob)
        ob, reward, done, _ = env.step(action)
        if done:
            break
