# File: q_learning.py 
# Author(s): Rishikesh Vaishnav
# Created: 29/06/2018

import logging
import os
import numpy as np;
import matplotlib.pyplot as plt;
from labellines import labelLine, labelLines;

import gym

# - graph parameters -
plt.rc('text', usetex=True);

ALPHA_COLOR_MULT = 0.3;
# - 

# Policy agent with various utility functions for use in emulation/training.
class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space;
        self.num_actions = action_space.n;

    def act(self, state, theta):
        action, _ = self.select_action(state, theta);
        return action;

    def act_eps(self, state, theta, epsilon):
        best_action = self.act(state, theta);
        rand_action = np.random.choice(a = np.array(range(self.num_actions)));

        action = np.random.choice(a = np.array([best_action, rand_action]), \
            p = [1 - epsilon, epsilon]);

        return action;

    # Selects an action using the given paramerized state-value function and 
    # current state.
    def select_action(self, state, theta):
        action_values = self.action_values(state, theta);
        action = np.argmax(action_values);
        value = action_values[action];

        return action, value;

    # Gets a vector representation of the given state and action.
    def vector(self, state, action):
        # set vector to be sets of states, with one set
        # activated, depending on the action
        vector = np.zeros((state.size * self.num_actions,));
        active_start_ind = action * state.size;
        active_end_ind = active_start_ind + state.size;
        vector[active_start_ind:active_end_ind] = state;

        return vector;

    # Gets the action-values using the given paramerized action-value function
    # and current state.
    def action_values(self, state, theta):
        # to store list of linearly mapped values for each action, with indices
        # directly corresponding to the action
        map_vals = np.array([0.0] * self.num_actions);

        # - set map_vals -
        for action in range(self.num_actions):
            map_vals[action] = self.action_value(state, action, theta);
        # -

        return map_vals;

    # Gets the action-value of the given state and action, using the given
    # parameters.
    def action_value(self, state, action, theta):
        vector = self.vector(state, action);
        return np.dot(vector, theta);

# - initialize training -
env = gym.make('CartPole-v0')
agent = Agent(env.action_space);

MAX_EPISODES = 400;
RUNS = 10;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
epsilon = 0.05;
alphas = [0.001, 0.01, 0.1];
# -

# Performs a training run using the given learning rate, returning an ordered 
# list of episode lengths.
def Q_training_run(alpha):
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    theta = np.zeros((env.observation_space.shape[0] \
            * env.action_space.n,));

    # continue training until cutoff
    for ep_i in range(MAX_EPISODES):
        # - run training episode -
        state = env.reset()

        for _ in range(MAX_STEPS):
            # optional - render episodes
            #env.render();

            orig_state = state;

            action = agent.act_eps(state, theta, epsilon);

            state, reward, done, _ = env.step(action);

            theta += alpha * \
            (reward + (gamma * agent.select_action(state, theta)[1]) \
            - agent.action_value(orig_state, action, theta)) \
            * agent.vector(orig_state, action);

            if done:
                break;

        alpha = alpha / 1.1;
        # -

        # - run evaluation episode -
        state = env.reset()

        t = 0;

        for t in range(MAX_STEPS):
            # optional - render episodes
            #env.render();

            action = agent.act(state, theta);

            state, _, done, _ = env.step(action);

            if done:
                break;

        # total time of episode
        T = t + 1;

        #print("\tEpisode Length: " + str(T));

        ep_lengths[ep_i] += T;
        # -


    return ep_lengths;


# mapping from alpha to episode length sequence
alpha_to_ep_lengths = {};

for alpha in alphas:
    print("Alpha: " + str(alpha));

    # sequence of episode lengths, totaled over all runs
    ep_lengths_tot = np.zeros((MAX_EPISODES,));

    for run_i in range(RUNS):
        print("\trun: " + str(run_i + 1) + "/" + str(RUNS));

        ep_lengths_tot = ep_lengths_tot + Q_training_run(alpha);

    alpha_to_ep_lengths[alpha] = ep_lengths_tot / RUNS;

alpha_i = 0;
for alpha in alphas:
    ep_lengths = alpha_to_ep_lengths[alpha];
    x = np.array(range(1, len(ep_lengths) + 1)).astype(float);
    y = ep_lengths;
    #color = (np.random.rand(), np.random.rand(), np.random.rand());
    color = (alpha_i * ALPHA_COLOR_MULT, alpha_i * ALPHA_COLOR_MULT, \
            alpha_i * ALPHA_COLOR_MULT);
    plt.plot(x, y, linestyle='-', color=color, label="$\\alpha =$ " \
            + str(alpha));

    alpha_i += 1;

plt.ylabel("Average Target Policy Episode Length (" + str(RUNS) + " Runs)");
plt.xlabel("Episode");
plt.title("Q Learning Function Approximator Results");

labelLines(plt.gca().get_lines(),zorder=2.5)

plt.savefig("Q_agent.pgf");

plt.show();
