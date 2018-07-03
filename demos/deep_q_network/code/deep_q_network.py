# File: deep_q_network.py 
# Author(s): Rishikesh Vaishnav
# Created: 29/06/2018
import numpy as np;
import random;
import matplotlib.pyplot as plt;
from labellines import labelLine, labelLines;

import gym

# - graph parameters -
plt.rc('text', usetex=True);

ALPHA_COLOR_MULT = 0.3;
# - 

# - initialize training -
env = gym.make('CartPole-v0')
num_observations = env.observation_space.shape[0];
num_actions = env.action_space.n;

MAX_EPISODES = 800;
RUNS = 1;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
alphas = [0.001, 0.01, 0.1];
# TODO? adjust
epsilon = 0.1;
C = 5;
replay_count = 3;
# -

# - model parameters -
hidden_layer_size = 8;
# -

# - model functions -
def action_value(state, action, theta):
    T = action_values(state, action, theta):

    # TODO optimize for not calculating all T;
    return T[action];

def action_values(state, action, theta):
    # TODO ensure state is a 1d vector
    print("state shape (should be 1d): " + str(state.shape));
    # hidden layer values
    Z = sigma(np.matmul(alpha_matrix(theta), state[:, None]));
    # TODO ensure Z is a 1d vector
    print("Z shape (should be 1d): " + str(Z.shape));

    # return action values
    return np.matmul(beta_matrix(theta), Z[:, None]);

# TODO action_value_grad()

def alpha_matrix(theta):
    alpha_mat = ab_matrix(theta, 0, num_observations, hidden_layer_size);

def beta_matrix(theta):
    beta_mat = ab_matrix(theta, hidden_layer_size * (num_observations + 1),\
        hidden_layer_size, num_actions);

def ab_matrix(theta, offset, in_layer_size, out_layer_size):
    ab_mat = np.zeros(out_layer_size, in_layer_size + 1);
    for i in range(out_layer_size):
        start_ind = offset + i * (in_layer_size + 1);
        end_ind = start_ind + (in_layer_size + 1);
        ab_mat[i, :] = theta[start_ind:end_ind];
    return ab_mat;


def sigma(mapped_vals):
    return 1 / (1 + e ** (-mapped_vals));
    
# -

# Performs a training run using the given learning rate, returning an ordered 
# list of episode lengths.
def DQN_training_run(alpha):
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    # TODO initialize theta
    theta = 

    # TODO initialize target theta
    target_theta = theta;

    # number of time steps since the last update to the target theta
    target_outdate_count = 0;

    # TODO initialize replay memory
    replay_mem = [];

    # continue training until cutoff
    for ep_i in range(MAX_EPISODES):
        # - run training episode -
        state = env.reset()

        for _ in range(MAX_STEPS):
            # optional - render episodes
            env.render();

            # save original state
            orig_state = state;
            
            # - determine action -
            # do random action to ensure exploration
            if np.random.rand() < epsilon:
                action = env.action_space.sample();
            # do best action
            else:
                action = np.argmax(action_values(state, theta));
            # -

            # perform action and record results
            state, reward, done, _ = env.step(action);

            # add transition to replay memory
            replay_mem.append((orig_state, action, reward, state, done));

            for _ in range(replay_count):
                # get random sample
                s_orig_state, s_action, s_reward, s_state, s_done = \
                    random.choice(replay_mem);

                # get target
                if not s_done:
                    action_val = np.max(action_values(s_state, target_theta));
                    target = s_reward + (gamma * action_val);
                else:
                    target = s_reward;

                # - update theta -
                delta = target - action_value(s_orig_state, s_action, theta);

                theta = theta + alpha * 2 * delta * action_value_grad(\
                    s_orig_state, s_action, theta);
                # -
            
            # - update target theta -
            target_outdate_count += 1;
            if target_outdate_count == C:
                target_theta = theta;
                target_outdate_count = 0;
            # -
            
            if done:
                break;
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
