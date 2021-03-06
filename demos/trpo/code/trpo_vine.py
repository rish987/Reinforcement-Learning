# File: trpo.py 
# Author(s): Rishikesh Vaishnav
# Created: 10/07/2018

import numpy as np;
import gym;
import random;
from decimal import Decimal;
from scipy.stats import entropy;
from scipy.optimize import minimize;
from scipy.optimize import BFGS;
from scipy.optimize import SR1;
from scipy.optimize import NonlinearConstraint;
import matplotlib.pyplot as plt;
import pickle;

RAND_SEED = 0;

np.random.seed(RAND_SEED);
random.seed(RAND_SEED);

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
        map_vals = np.array([Decimal(0.0)] * self.num_actions, \
                dtype=np.dtype(Decimal));

        # - set map_vals -
        for action in range(self.num_actions):
            # set vector to be sets of states, with one set
            # activated, depending on the action
            vector = self.vector(state, action);

            map_vals[action] = Decimal(np.dot(vector.astype(np.dtype(Decimal)), \
                theta.astype(np.dtype(Decimal))));
        # -


        # exponentiate the map vals
        map_vals_exp = np.exp(map_vals);

        # get softmax action probabilities
        action_probs = ((map_vals_exp) / np.sum(map_vals_exp)).astype(float);

        return map_vals_exp, action_probs;

# - initialize training -
env = gym.make('CartPole-v0')
env.seed(RAND_SEED);
agent = Agent(env.action_space)

NUM_ITER = 15;
RUNS = 5;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
VINE_TRAJ = 20;
VINE_SAMP = 200;
TRAJ_PER_ITER = VINE_TRAJ + (VINE_SAMP * env.action_space.n);
TRAJ_PER_VINE_TRAJ = int(TRAJ_PER_ITER / VINE_TRAJ);
delta = 0.01;
MAX_ITERS = 500;

# paramerization of policy to be deployed at vine sample states
# CURRENT POLICY: equal likelihood policy
theta_vine = np.zeros((env.observation_space.shape[0] \
        * env.action_space.n,));

# number of episodes in a single run
MAX_EPISODES = NUM_ITER * TRAJ_PER_ITER;
# -

# Calculate KL Divergence of two policies under a certain state.
def kl_divergence(action_probs_old, theta2, state):
    action_probs1 = action_probs_old[tuple(state.tolist())];
    action_probs2 = agent.action_probs(state, theta2)[1];

    D = entropy(action_probs1, action_probs2);
#    D = 0;
#
#    for action in range(env.action_space.n):
#        prob1 = action_probs1[action];
#        prob2 = action_probs2[action];
#	
#        if prob2 > 1.0e-12:
#            D +=  prob1 * np.log(prob1 / prob2);

    if D < 0:
        if D < -1.0e-5:
            print("WARNING: negative KL-divergence.");
            print(action_probs1[0] + action_probs1[1]);
            print(action_probs2[0] + action_probs2[1]);
            print(D);
        return 0;

    return D;

# Calculate value of constraint function.
def constraint(S, theta, action_probs_old):
    D = 0;
    for _, state, _ in S:
        D += kl_divergence(action_probs_old, theta, state);

    return D / len(S);

# Calculate value of objective function.
def objective(S, theta):
    L = 0;
    for G_div_p, state, action in S:
        L += G_div_p * agent.action_probs(state, theta)[1][action];

    return L;

def checker(a, b):
    print("\t\tParameters: \n\t\t" + str(a[0:4]) + "\n\t\t" + str(a[4:]));

def run_episode(theta, s_a=None):

    t = 0;

    # list of rewards
    rewards = [];
    # list of states
    states = [];
    # list of actions
    actions = [];

    if s_a == None:
        state = env.reset()
        action = agent.act(state, theta);
    else:
        state = s_a[0];
        action = s_a[1];

    for t in range(MAX_STEPS):
        # optional - render episodes
        #env.render();

        # save this time's state and action
        states.append(state);
        actions.append(action);

        state, reward, done, _ = env.step(action);

        # save the next time's reward
        rewards.append(reward);

        if done:
            break;

        action = agent.act(state, theta);

    return rewards, states, actions, t;

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

        # to store all states along all trajectories in the case of vine
        all_states = [];

        # go through all super trajectories
        for traj_i in range(VINE_TRAJ): 
            # - run episode -
            rewards, states, actions, t = run_episode(theta);

            # total time of episode
            T = t + 1;

            # add states to list
            all_states += states;

            ep_lengths_off = (iter_i * TRAJ_PER_ITER) \
                    + (traj_i * TRAJ_PER_VINE_TRAJ);

            # duplicate this result for the next VINE_SAMP * 2 episodes,
            # rather than using the results from the vine trajectories
            for i in range(TRAJ_PER_VINE_TRAJ):
                ep_lengths[i + ep_lengths_off] += T;

            print('\t\tepisode length: ' + str(T));
        for _ in range(VINE_SAMP):
            # sample a state
            s_state = random.choice(all_states);

            _, probs = agent.action_probs(s_state, theta_vine);

            for _ in range(env.action_space.n):
                # sample an action according to the vine policy
                s_action = np.random.choice(a = range(env.action_space.n), \
                        p = probs);

                s_a = (s_state, s_action);

                env.reset();
                env.state = s_state;
                # produce a trajectory from this state and action
                rewards, states, actions, t = run_episode(theta, s_a);

                T = t + 1;

                # remaining discounted reward from this time
                G = 0;

                # go backwards through episode
                for t in range(T)[::-1]:
                    G = rewards[t] + gamma * G;

                G_div_p = G / probs[s_action];
                S.append((G_div_p, s_state, s_action));

        # - update parameters -
        # this is not the last iteration, so updated parameters would be useful
        if iter_i < (NUM_ITER - 1):
            action_probs_old = {};
            for _, state, _ in S:
                action_probs_old[tuple(state.tolist())] = \
                    agent.action_probs(state, theta)[1];
            constraint_var = NonlinearConstraint(lambda x:constraint(S, x,\
                action_probs_old), -np.inf, delta, jac='2-point', hess=BFGS());
            theta = minimize(lambda x:(-1 * objective(S, x)), theta, \
                method='trust-constr',  jac="2-point",\
                hess=SR1(), constraints=[constraint_var],\
                options={'verbose': 1, 'maxiter': MAX_ITERS},\
                tol=1.0e-2,\
                callback=checker).x;
        # -

    return ep_lengths;

# - perform runs -
# sequence of episode lengths, totaled over all RUNS
ep_lengths = np.zeros((RUNS, MAX_EPISODES));

for run_i in range(RUNS):
    print("run: " + str(run_i + 1) + "/" + str(RUNS));

    ep_lengths[run_i, :] = TRPO_training_run();

with open("vine_ep_lengths.dat", 'wb') as file:
    pickle.dump(ep_lengths, file);

ep_lengths_avg = np.sum(ep_lengths, axis=0).astype(float) / RUNS;
# - 

# - plot results -
x = np.array(range(1, len(ep_lengths_avg) + 1)).astype(float);
y = ep_lengths_avg;
color = (0.0, 0.0, 0.0);
plt.plot(x, y, linestyle='-', color=color);

plt.ylabel("Average Episode Length (" + str(RUNS) + " Runs)");
plt.xlabel("Episode");
plt.title("Trust Region Policy Optimization Results");

plt.savefig("TRPO_agent.pgf");

plt.show();
# -
