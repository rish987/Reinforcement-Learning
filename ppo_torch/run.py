# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed, Dataset
from misc_utils import RO_EP_LEN, RO_EP_RET, RO_OB, RO_AC, RO_ADV_GL,\
    RO_VAL_GL, GRAPH_OUT
from ppo_model import PPOModel
from rollout import get_rollout
import matplotlib.pyplot as plt;
plt.rc('text', usetex=True)

# - hyperparameters -
# -- neural network parameters --
# number of nodes in each hidden layer
g_hidden_layer_size = 64
# number of hidden layers
g_num_hidden_layers = 2
# --

# -- run parameters --
# number of timesteps to train over
g_num_timesteps = 500000
# number of timesteps in a single rollout (simulated trajectory with fixed
# parameters)
g_timesteps_per_rollout = 2048 * 6
# random seed
g_seed = 0
# -- 

# epsilon as described by Schulman et. al.
g_clip_param = 0.2

# -- SGD parameters --
# number of training epochs per run
g_num_epochs = 20
# adam learning rate
g_alpha = 3e-4
# number of randomly selected timesteps to use in a single parameter update
g_batch_size = 1024
# --

# -- GAE parameters --
# gamma and lambda factors used in Generalized Advantage Estimation (Schulman
# et. al.) to trade off between variance and bias in return/advantage
# approximation
g_gamma = 0.99
g_lambda_ = 0.95
# -- 
# - 

# TODO replace with passed-in environment
g_env_name = "InvertedPendulum-v2"

"""
Trains a PPO agent according to given parameters and reports results.
"""
def train(hidden_layer_size, num_hidden_layers, num_timesteps, \
    timesteps_per_rollout, seed, clip_param, num_epochs, alpha, \
    batch_size, gamma, lambda_, env_name):
    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env.seed(seed)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    model = PPOModel(env, num_hidden_layers, hidden_layer_size, alpha,\
            clip_param)

    # total number of timesteps trained so far
    timesteps = 0

    # generator for getting rollouts
    rollout_gen = get_rollout(env, model, timesteps_per_rollout, gamma, lambda_)
    # - 

    # - initialize graph data -
    # to store negated changes in standard deviations at each iteration
    stds = [];

    # to store the average number of upclips (r_t > 1 + epsilon) and downclips
    # (r_t < 1 - epsilon) taken over all batches at each iteration
    avg_num_upclips = [];
    avg_num_downclips = [];
    # - 

    # - training -
    # continue training until timestep limit is reached
    while (timesteps < num_timesteps):
        # - SGD setup - 
        model.update_cpu_networks()

        # get a rollout under this model for training
        rollout = rollout_gen.__next__()
        
        # update old policy function to new policy function
        model.update_old_pol()

        # place data into dataset that will shuffle and batch them for training
        data = Dataset({RO_OB:rollout[RO_OB], RO_AC:rollout[RO_AC],\
            RO_ADV_GL:rollout[RO_ADV_GL], RO_VAL_GL:rollout[RO_VAL_GL]})

        # linearly decrease learning rate TODO delete?
#        alpha_decay_factor = \
#                max(1.0 - float(timesteps) / num_timesteps, 0)
        # - 

        # number of upclips and downclips at each batch
        num_upclips = []
        num_downclips = []

        # - SGD training -
        # go through all epochs
        for e in range(num_epochs):
            # go through all randomly organized batches
            for batch in data.iterate_once(batch_size):
                # get gradient
                num_upclips_new, num_downclips_new =\
                    model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL])
                num_upclips.append(num_upclips_new)
                num_downclips.append(num_downclips_new)
        # - 

        avg_num_upclips.append(np.mean(num_upclips))
        avg_num_downclips.append(np.mean(num_downclips))

        # standard deviation after update
        std = model.policy_net.std().item();

        # save standard deviation
        stds.append(std)

        # update total timesteps traveled so far
        timesteps += np.sum(rollout[RO_EP_LEN])
        #print(avg_num_upclips[-1], avg_num_downclips[-1], stds[-1])
        print("Time Elapsed: {0}; Average Reward: {1}".format(timesteps, 
            np.mean(rollout[RO_EP_LEN][-100:])))
    # - 

    graph_data = {}
    graph_data[GD_STD] = np.array(stds)
    graph_data[GD_AVG_NUM_UPCLIPS] = np.array(avg_num_upclips)
    graph_data[GD_AVG_NUM_DOWNCLIPS] = np.array(avg_num_downclips)

    return graph_data

def global_run_seed(seed):
    return train(hidden_layer_size=g_hidden_layer_size, \
        num_hidden_layers=g_num_hidden_layers, num_timesteps=g_num_timesteps, \
        timesteps_per_rollout=g_timesteps_per_rollout, seed=seed, \
        clip_param=g_clip_param, num_epochs=g_num_epochs, alpha=g_alpha, \
        batch_size=g_batch_size, gamma=g_gamma, lambda_=g_lambda_, \
        env_name=g_env_name)

def get_average_list_arr(list_arr):
    arr_len = min([len(arr) for arr in list_arr]);
    list_arr_cropped = [arr[:arr_len] for arr in list_arr]
    return np.average(np.array(list_arr_cropped), axis=0)

def graph_stds_and_clips(stds, avg_num_upclips, avg_num_downclips):
    iterations = np.arange(stds.shape[0]) + 1

    plt.figure()
    plt.subplot(211)
    plt.ylabel("count")
    plt.xlabel("Number of Iterations")
    plt.title("Clipping Behavior")

    plt.plot(iterations, avg_num_upclips, linestyle='-', \
            color=(0.5, 0.5, 0.5), label='$|\{r_t \mid r_t > 1 + \epsilon\}|$')
    plt.plot(iterations, avg_num_downclips, linestyle='-', \
            color=(0.7, 0.7, 0.7), label='$|\{r_t \mid r_t < 1 - \epsilon\}|$')

    plt.legend()
    
    plt.subplot(212)
    plt.ylabel("$\sigma$")
    plt.xlabel("Number of Iterations")
    plt.title("Standard Deviation")
    plt.plot(iterations, stds, linestyle='-', color=(0.0, 0.0, 0.0),\
            label='$\sigma$')

    plt.legend()

    plt.tight_layout()

    plt.savefig(GRAPH_OUT)
    plt.show()

def stds_and_clips_run():
    all_stds = []
    all_avg_num_upclips = []
    all_avg_num_downclips = []
    for seed in range(8):
        graph_data = global_run_seed(seed)
        stds = graph_data[GD_STD]
        avg_num_upclips = graph_data[GD_AVG_NUM_UPCLIPS]
        avg_num_downclips = graph_data[GD_AVG_NUM_DOWNCLIPS]
        all_stds.append(stds)
        all_avg_num_upclips.append(avg_num_upclips)
        all_avg_num_downclips.append(avg_num_downclips)

    stds_avg = get_average_list_arr(all_stds)
    avg_num_upclips_avg = get_average_list_arr(all_avg_num_upclips)
    avg_num_downclips_avg = get_average_list_arr(all_avg_num_downclips)
    
    graph_stds_and_clips(stds_avg, avg_num_upclips_avg,\
            avg_num_downclips_avg)

def main():
    stds_and_clips_run()

if __name__ == '__main__':
    main()
