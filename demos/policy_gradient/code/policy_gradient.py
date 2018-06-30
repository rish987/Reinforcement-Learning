# File: policy_gradient.py 
# Author(s): Rishikesh Vaishnav
# Created: 29/06/2018
import logging
import os
import numpy as np;

import gym

# Monte Carlo Policy Gradient agent
class MCPG_agent(object):
    def __init__(self, action_space):
        self.action_space = action_space;
        self.num_actions = action_space.n;

    def act(self, state, theta):
        return self.select_action(state, theta);

    # Selects the an action using the given paramerized policy and current
    # state.
    def select_action(self, state, theta):
        _, probs = self.action_probs(state, theta);
        action = np.random.choice(a = np.array([0, 1]), p = probs);

        return action;

    # Gets a vector representation of the given state and action.
    def vector(self, state, action):
        # set vector to be sets of states, with one set
        # activated, depending on the action
        vector = np.zeros((state.size * self.num_actions,)).astype(float);
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

        # TODO
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

max_episodes = 100;
max_steps = 200;

reward = 0
done = False
theta = np.random.rand(env.observation_space.shape[0] * env.action_space.n);

# TODO adjust
# training parameters
gamma = 1.0;
alphas = [0.01 * n for n in range(1, 31)];
# -

# mapping from alpha to episode length sequence
alpha_to_ep_lengths = {};

for alpha in alphas:
    # sequence of episode lengths
    ep_lengths = [];

    # continue training until cutoff
    for _ in range(max_episodes):
        # - run episode -
        state = env.reset()

        t = 0;

        # list of rewards
        R = [];
        # list of states
        S = [];
        # list of actions
        A = [];

        for t in range(max_steps):
            # optional - render episodes
            #env.render();

            action = agent.act(state, theta);

            # save this time's state and action
            S.append(state);
            A.append(action);

            state, reward, done, _ = env.step(action);

            # save the next time's reward
            R.append(reward);

            if done:
                print("here");
                break;

        # total time of episode
        T = t + 1;

        ep_lengths.append(T);

        #print("Episode length: " + str(T));
        # -

        # - update parameters -
        for t in range(T):
            # get reward from this time to the end
            G = 0;
            discount_factor = 1;
            for r in R[t:]:
                G += discount_factor * r;
                discount_factor *= gamma;

            theta += alpha * G * agent.grad_ln_probs(S[t], A[t], theta);
        # -

    alpha_to_ep_lengths[alpha] = ep_lengths;
