# File: deep_q_network_keras.py 
# Author(s): Rishikesh Vaishnav
# Created: 03/07/2018
import numpy as np;
import random;
import matplotlib.pyplot as plt;
from labellines import labelLine, labelLines;

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

MAX_EPISODES = 100;
RUNS = 1;
MAX_STEPS = 200;

# training parameters
gamma = 1.0;
alphas = [0.001, 0.01, 0.1];
# TODO? adjust
C = 3;
INIT_EPSILON = 1.0;
MIN_EPSILON = 0.01;
EPSILON_DECAY = 0.995;
replay_count = 30;
replay_size = 50;
# -

# - model parameters -
hidden_layer_size = 24;
# -

# Performs a training run using the given learning rate, returning an ordered 
# list of episode lengths.
def DQN_training_run(alpha):
    # sequence of episode lengths for this run
    ep_lengths = np.zeros((MAX_EPISODES,));

    reward = 0;
    done = False;
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=num_observations, \
        activation='relu'))
    model.add(Dense(hidden_layer_size, activation='relu'));
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=alpha))

    target_model = Sequential();
    target_model.add(Dense(hidden_layer_size, input_dim=num_observations, \
        activation='relu'))
    target_model.add(Dense(hidden_layer_size, activation='relu'));
    target_model.add(Dense(num_actions, activation='linear'))
    target_model.compile(loss='mse', optimizer=Adam(lr=alpha))
    target_model.set_weights(model.get_weights());

    epsilon = INIT_EPSILON;

    # number of time steps since the last update to the target model
    target_outdate_count = 0;

    # initialize replay memory
    replay_mem = [];

    # continue training until cutoff
    for ep_i in range(MAX_EPISODES):
        print("\t\tepisode " + str(ep_i + 1) + "/" + str(MAX_EPISODES));

        # - run training episode -
        state = env.reset();

        for t in range(MAX_STEPS):
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
                action = np.argmax(model.predict(state[None, :])[0]);
            # -

            # perform action and record results
            state, reward, done, _ = env.step(action);
            reward = -10 if (done and t < 199) else reward;

            # add transition to replay memory
            replay_mem.append((orig_state, action, reward, \
                state, done));

            if len(replay_mem) > replay_size:
                del replay_mem[0];

            for _ in range(replay_count):
                # get random sample
                s_orig_state, s_action, s_reward, s_state, s_done = \
                    random.choice(replay_mem);

                # get target
                if not s_done:
                    action_val = np.amax(target_model.predict(s_state[None, :]\
                        )[0]);
                    target = s_reward + (gamma * action_val);
                else:
                    target = s_reward;

                # - update model -
                target_f = model.predict(s_orig_state[None, :])
                target_f[0][s_action] = target
                model.fit(s_orig_state[None, :], target_f, epochs=1, verbose=0)
                # -
            
            # - update target model -
            target_outdate_count += 1;
            if target_outdate_count == C:
                target_model.set_weights(model.get_weights());
                target_outdate_count = 0;
            # -

            alpha /= 1.0001;
            
            if done:
                break;
        # -
        if epsilon > MIN_EPSILON:
            epsilon = epsilon * EPSILON_DECAY;

        # - run evaluation episode -
        state = env.reset();

        t = 0;

        for t in range(MAX_STEPS):
            # optional - render episodes
            #env.render();

            action = np.argmax(model.predict(state[None, :])[0]);

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

plt.savefig("DQN_Keras.pgf");

plt.show();
# -
