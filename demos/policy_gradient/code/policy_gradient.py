# File: policy_gradient.py 
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
class MCPG_agent(object):
    def __init__(self, action_space):
        self.action_space = action_space;
        self.num_actions = action_space.n;

    def act(self, state, theta):
        return self.select_action(state, theta);

    # Selects an action using the given paramerized policy and current
    # state.
    def select_action(self, state, theta):
        _, probs = self.action_probs(state, theta);
        action = np.random.choice(a = np.array([0, 1]), p = probs);

        return action;

    # Gets a vector representation of the given state and action.
    def vector(self, state, action):
        # set vector to be sets of states, with one set
        # activated, depending on the action
        vector = np.zeros((state.size * self.num_actions,));
        active_start_ind = action * state.size;
        active_end_ind = active_start_ind + state.size;
        vector[active_start_ind:active_end_ind] = state;

        return vector;

    # Gets the action probabilities using the given paramerized policy and
    # current state.
    def action_probs(self, state, theta):
        # to store list of linearly mapped values for each action, with indices
        # directly corresponding to the action
        map_vals = np.array([0.0] * self.num_actions);

        # - set map_vals -
        for action in range(self.num_actions):
            # set vector to be sets of states, with one set
            # activated, depending on the action
            vector = self.vector(state, action);

            map_vals[action] = np.dot(vector, theta);
        # -

        # exponentiate the map vals
        map_vals_exp = np.exp(map_vals);

        # get softmax action probabilities
        action_probs = (map_vals_exp) / np.sum(map_vals_exp);

        return map_vals_exp, action_probs;

    # Gets the gradient of the ln of the probability, evaluated at the current
    # state and action.
    def grad_ln_probs(self, state, action, theta):
        map_vals_exp, action_probs = self.action_probs(state, theta);
        return self.grad_probs(map_vals_exp, state, action, theta) / \
            action_probs[action];

    # Gets the gradient of the probability, evaluated at the current state and
    # action.
    def grad_probs(self, map_vals_exp, state, action, theta):
        this_map_val_exp = map_vals_exp[action];
        sum_map_vals_exp = np.sum(map_vals_exp);
        weighted_vectors = np.array([self.vector(state, a) for a in \
            range(self.num_actions)]).T * map_vals_exp;

        term_1 = self.vector(state, action) * sum_map_vals_exp;
        term_2 = np.sum(weighted_vectors, axis = 1);

        grad = (this_map_val_exp / (sum_map_vals_exp ** 2)) * (term_1 - term_2);

        return grad;

# - initialize training -
env = gym.make('CartPole-v0')
agent = MCPG_agent(env.action_space)

MAX_EPISODES = 400;
RUNS = 10;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
alphas = [0.001, 0.01, 0.1];
num_alphas = 2;
# -

# Performs a training run using the given learning rate, returning an ordered 
# list of episode lengths.
def MCPG_training_run(alpha):
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    theta = np.zeros((env.observation_space.shape[0] \
            * env.action_space.n,));

    # continue training until cutoff
    for ep_i in range(MAX_EPISODES):
        # - run episode -
        state = env.reset()

        t = 0;

        # list of rewards
        R = [];
        # list of states
        S = [];
        # list of actions
        A = [];

        for t in range(MAX_STEPS):
            # optional - render episodes
            # env.render();

            action = agent.act(state, theta);

            # save this time's state and action
            S.append(state);
            A.append(action);

            state, reward, done, _ = env.step(action);

            # save the next time's reward
            R.append(reward);

            if done:
                break;

        # total time of episode
        T = t + 1;

        ep_lengths[ep_i] += T;

        # -

        # - update parameters -
        # to store reward from this time to the end
        G = 0;

        for t in reversed(range(T)):
            # get reward from this time to the end
            G = (gamma * G) + R[t];

            theta += alpha * G * agent.grad_ln_probs(S[t], A[t], theta);
        # -

    return ep_lengths;


# mapping from alpha to episode length sequence
alpha_to_ep_lengths = {};

for alpha in alphas:
    print("Alpha: " + str(alpha));

    # sequence of episode lengths, totaled over all RUNS
    ep_lengths_tot = np.zeros((MAX_EPISODES,));

    for run_i in range(RUNS):
        print("\trun: " + str(run_i + 1) + "/" + str(RUNS));

        ep_lengths_tot = ep_lengths_tot + MCPG_training_run(alpha);

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

plt.ylabel("Average Episode Length (" + str(RUNS) + " Runs)");
plt.xlabel("Episode");
plt.title("Monte Carlo Policy Gradient Results");

labelLines(plt.gca().get_lines(),zorder=2.5)

plt.savefig("MCPG_agent.pgf");

plt.show();
