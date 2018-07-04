# File: deep_q_network.py 
# Author(s): Rishikesh Vaishnav
# Created: 29/06/2018
import numpy as np;
import random;
import matplotlib.pyplot as plt;
from labellines import labelLine, labelLines;

import gym

np.random.seed(0);
random.seed(0);

# - graph parameters -
plt.rc('text', usetex=True);

ALPHA_COLOR_MULT = 0.3;
# - 

# - initialize training -
env = gym.make('CartPole-v0')
env.seed(0);
num_observations = env.observation_space.shape[0];
num_actions = env.action_space.n;

MAX_EPISODES = 2000;
RUNS = 1;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
#alphas = [0.001, 0.01, 0.1];
alphas = [0.0025];#, 0.01, 0.1];
# TODO? adjust
C = 100;
replay_count = 30;
replay_size = 50;
# -

# - model parameters -
hidden_layer_size = 20;
# -

# - model functions -
def action_values(state, theta):
    pre_Z = np.matmul(alpha_matrix(theta), np.insert(state, 0, 1)[:, None]);
    # hidden layer values
    Z = sigma(pre_Z);

    # return action values
    return np.matmul(beta_matrix(theta), np.insert(Z, 0, 1)[:, None]), Z, pre_Z;

def action_value_grad(state, action, theta, sigma_vals, map_vals):
    beta_grad_mat = np.zeros((num_actions, hidden_layer_size + 1));
    beta_grad_mat[action, 0] = 1;
    beta_grad_mat[action, 1:] = sigma_vals;

    col_multiplier = np.repeat(\
        np.multiply(beta_matrix(theta)[action, 1:][:, None], \
        sigma_grad(map_vals)\
        [:, None]),\
        num_observations + 1, axis=1);

    row_multiplier = np.repeat(\
        np.insert(state, 0, 1)[None, :],\
        hidden_layer_size, axis=0);

    alpha_grad_mat = np.multiply(col_multiplier, row_multiplier);

    return ab_matrix_to_theta_full(alpha_grad_mat, beta_grad_mat);

def alpha_matrix(theta):
    return ab_matrix(theta, 0, num_observations, hidden_layer_size);

def beta_matrix(theta):
    return ab_matrix(theta, hidden_layer_size * (num_observations + 1),\
        hidden_layer_size, num_actions);

def ab_matrix(theta, offset, in_layer_size, out_layer_size):
    ab_mat = np.reshape(theta[offset:offset + out_layer_size *\
            (in_layer_size + 1)], (out_layer_size, in_layer_size + 1));
    return ab_mat;

def ab_matrix_to_theta_full(alpha, beta):
    return np.concatenate((alpha.flatten(), beta.flatten()));

def sigma(mapped_vals):
    return (1 / (1 + np.exp(-mapped_vals))).flatten();

def sigma_grad(mapped_vals):
    return (np.exp(-mapped_vals) / ((1 + np.exp(-mapped_vals)) ** 2)).flatten();
# -

# Performs a training run using the given learning rate, returning an ordered 
# list of episode lengths.
def DQN_training_run(alpha):
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    theta = np.zeros(\
        ((num_observations + 1) * hidden_layer_size + \
        (hidden_layer_size + 1) * num_actions, ) \
        );
    target_theta = theta;

    epsilon = 0.5;

    # number of time steps since the last update to the target theta
    target_outdate_count = 0;

    # initialize replay memory
    replay_mem = [];

    # continue training until cutoff
    for ep_i in range(MAX_EPISODES):
        print("\t\tepisode " + str(ep_i + 1) + "/" + str(MAX_EPISODES));

        # - run training episode -
        state = env.reset()

        for _ in range(MAX_STEPS):
            # optional - render episodes
            #env.render();

            # save original state
            orig_state = state;
            
            # - determine action -
            # do random action to ensure exploration
            if np.random.rand() < epsilon:
                action = env.action_space.sample();
            # do best action
            else:
                action = np.argmax(action_values(state, theta)[0]);
            # -

            # perform action and record results
            state, reward, done, _ = env.step(action);

            # add transition to replay memory
            replay_mem.append((orig_state, action, reward, state, done));

            if len(replay_mem) > replay_size:
                del replay_mem[0];

            for _ in range(replay_count):
                # get random sample
                s_orig_state, s_action, s_reward, s_state, s_done = \
                    random.choice(replay_mem);

                # get target
                if not s_done:
                    action_val = np.max(action_values(s_state,\
                        target_theta)[0]);
                    target = s_reward + (gamma * action_val);
                else:
                    target = s_reward;

                # - update theta -
                action_vals, sigma_vals, map_vals =\
                    action_values(s_orig_state, theta);
                delta = target - action_vals[s_action];

                theta = theta + alpha * 2 * delta * action_value_grad(\
                    s_orig_state, s_action, theta, sigma_vals, map_vals);
                #print(theta);
                # -
            
            # - update target theta -
            target_outdate_count += 1;
            if target_outdate_count == C:
                target_theta = theta;
                target_outdate_count = 0;
            # -

            #epsilon -= epsilon * 0.0001;
            alpha /= 1.0001;
            
            if done:
                break;
        # -
        print(alpha);

        # - run evaluation episode -
        state = env.reset()

        t = 0;

        for t in range(MAX_STEPS):
            # optional - render episodes
            #env.render();

            action = np.argmax(action_values(state, theta)[0]);

            state, _, done, _ = env.step(action);

            if done:
                break;

        # total time of episode
        T = t + 1;

        print("\tEpisode Length: " + str(T));

        ep_lengths[ep_i] += T;
        # -

    return ep_lengths;

# mapping from alpha to episode length sequence
alpha_to_ep_lengths = {};

# - collect results -
for alpha in alphas:
    print("Alpha: " + str(alpha));

    # sequence of episode lengths, totaled over all runs
    ep_lengths_tot = np.zeros((MAX_EPISODES,));

    for run_i in range(RUNS):
        print("\trun: " + str(run_i + 1) + "/" + str(RUNS));

        ep_lengths_tot = ep_lengths_tot + DQN_training_run(alpha);

    alpha_to_ep_lengths[alpha] = ep_lengths_tot / RUNS;
# -

# - plot results -
alpha_i = 0;
for alpha in alphas:
    ep_lengths = alpha_to_ep_lengths[alpha];
    x = np.array(range(1, len(ep_lengths) + 1)).astype(float);
    y = ep_lengths;
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
# -
