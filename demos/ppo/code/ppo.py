# File: ppo.py
# Author(s): Rishikesh Vaishnav
# Created: 12/07/2018

import autograd.numpy as np;
import gym;
import random;
import matplotlib.pyplot as plt;
import pickle;
from autograd import grad;

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
        map_vals = np.array([0.0] * self.num_actions);

        vectors = np.zeros((self.num_actions, theta.shape[0]));

        # - set map_vals -
        for action in range(self.num_actions):
            # set vector to be sets of states, with one set
            # activated, depending on the action
            vectors[action, :] = self.vector(state, action);

        map_vals = np.dot(vectors, theta);
        # -

        # exponentiate the map vals
        map_vals_exp = np.exp(map_vals);

        # get softmax action probabilities
        action_probs = ((map_vals_exp) / np.sum(map_vals_exp));

        return map_vals_exp, action_probs;

# - initialize training -
env = gym.make('CartPole-v0')
env.seed(RAND_SEED);
agent = Agent(env.action_space)

NUM_ITER = 50;
RUNS = 5;
MAX_STEPS = 200;

# training parameters
alpha = 0.1;
gamma = 1.0;
NUM_TRAJ = 5;
epsilon = 0.02;
MAX_ITERS = 500;

# number of episodes in a single run
MAX_EPISODES = NUM_ITER * NUM_TRAJ;
# -

# Clips the given probability ratio according to the given epsilon.
def clip(r, eps):
    if (r > 1 + eps):
        return 1 + eps;
    elif (r < 1 - eps):
        return 1 - eps;

    return r;

# Calculate value of objective function.
def objective(theta, S):
    L = 0;
    for G, prob, state, action in S:
        r = (agent.action_probs(state, theta)[1][action] / prob);
        L += min(G * r, G * clip(r, epsilon));

    return L / len(S);

# Performs a training run, returning an ordered 
# list of episode lengths.
def PPO_training_run():
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

            ep_lengths[traj_i + (iter_i * NUM_TRAJ)] += T;

            print('\t\tepisode length: ' + str(T));

            # -

            # - calculate this S-tuple -
            # remaining discounted reward from this time
            G = 0;

            # go backwards through episode
            for t in range(T)[::-1]:
                G = rewards[t] + gamma * G;
                state = states[t];
                action = actions[t];
                prob = agent.action_probs(state, theta)[1][action];
                S.append((G, prob, state, action));
            # -

        # - update parameters -
        # this is not the last iteration, so updated parameters would be useful
        if iter_i < (NUM_ITER - 1):
            grad_objective = grad(lambda x: objective(x, S));
            theta += alpha * grad_objective(theta);
            print("\t\tParameters: \n\t\t" + \
                str(theta[0:4]) + "\n\t\t" + str(theta[4:]));
        # -

    return ep_lengths;

# - perform runs -
# sequence of episode lengths, totaled over all RUNS
ep_lengths = np.zeros((RUNS, MAX_EPISODES));

for run_i in range(RUNS):
    print("run: " + str(run_i + 1) + "/" + str(RUNS));

    ep_lengths[run_i, :] = PPO_training_run();

with open("single_path_ep_lengths.dat", 'wb') as file:
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
