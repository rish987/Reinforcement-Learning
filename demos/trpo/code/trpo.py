# File: trpo.py 
# Author(s): Rishikesh Vaishnav
# Created: 10/07/2018

import numpy as np;
import gym;
import random;
from scipy.optimize import minimize;
from scipy.optimize import BFGS;
from scipy.optimize import SR1;
from scipy.optimize import NonlinearConstraint;

np.random.seed(0);
random.seed(0);

# Policy agent with various utility functions for use in emulation/training.
class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space;
        self.num_actions = action_space.n;

    # Selects an action using the given paramerized policy and current
    # state.
    def act(self, state, theta):
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

# - initialize training -
env = gym.make('CartPole-v0')
env.seed(0);
agent = Agent(env.action_space)

NUM_ITER = 100;
RUNS = 10;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
NUM_TRAJ = 4;
delta = 0.01;

# number of episodes in a single run
MAX_EPISODES = NUM_ITER * NUM_TRAJ;
# -

# Calculate KL Divergence of two policies under a certain state.
def kl_divergence(theta1, theta2, state):
    action_probs1 = agent.action_probs(state, theta1)[1];
    action_probs2 = agent.action_probs(state, theta2)[1];

    D = 0;

    for action in range(env.action_space.n):
        prob1 = action_probs1[action];
        prob2 = action_probs2[action];
	
        if prob2 != 0:
            D +=  prob1 * np.log(prob1 / prob2);

    if D < -1.0e-5:
        print("WARNING: negative KL-divergence.");
        print(D);

    return D;

# Calculate value of constraint function.
def constraint(S, theta, theta_old):
    D = 0;
    for _, state, _ in S:
        D += kl_divergence(theta_old, theta, state);

    return D / len(S);

# Calculate value of objective function.
def objective(S, theta):
    L = 0;
    for G_div_p, state, action in S:
        L += G_div_p * agent.action_probs(state, theta)[1][action];

    return L;

def checker(a, b):
    print("Here: " + str(a));

# Performs a training run, returning an ordered 
# list of episode lengths.
def TRPO_training_run():
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    theta = np.zeros((env.observation_space.shape[0] \
            * env.action_space.n,));

    # continue training until cutoff
    for iter_i in range(NUM_ITER):
        print("\titeration: " + str(iter_i + 1) + "/" + str(NUM_ITER));
        # to store list of data tuples at each sample state and action
        S = [];

        # go through all trajectories
        for traj_i in range(NUM_TRAJ): 
            # - run episode -
            state = env.reset()

            t = 0;

            # list of rewards
            rewards = [];
            # list of states
            states = [];
            # list of actions
            actions = [];

            for t in range(MAX_STEPS):
                # optional - render episodes
                env.render();

                action = agent.act(state, theta);

                # save this time's state and action
                states.append(state);
                actions.append(action);

                state, reward, done, _ = env.step(action);

                # save the next time's reward
                rewards.append(reward);

                if done:
                    break;

            # total time of episode
            T = t + 1;

            ep_lengths[traj_i] += T;

            # -

            # - calculate this S-tuple -
            # remaining discounted reward from this time
            G = 0;

            # go backwards through episode
            for t in range(T)[::-1]:
                G = rewards[t] + gamma * G;
                state = states[t];
                action = actions[t];
                G_div_p = G / agent.action_probs(state, theta)[1][action];
                S.append((G_div_p, state, action));
            # -

        # - update parameters -
        constraint_var = NonlinearConstraint(lambda x:constraint(S, x, theta),\
            -np.inf, delta, jac='2-point', hess=BFGS());
        theta = minimize(lambda x:(-1 * objective(S, x)), theta, \
            method='trust-constr',  jac="2-point",\
            hess=SR1(), constraints=[constraint_var],\
            options={'verbose': 1},\
            tol=1.0e-2,\
            callback=checker).x;
        # -

    return ep_lengths;

# sequence of episode lengths, totaled over all RUNS
ep_lengths_tot = np.zeros((MAX_EPISODES,));

for run_i in range(RUNS):
    print("run: " + str(run_i + 1) + "/" + str(RUNS));

    ep_lengths_tot = ep_lengths_tot + TRPO_training_run();

alpha_to_ep_lengths[alpha] = ep_lengths_tot / RUNS;
