# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed, Dataset
from misc_utils import \
    RO_EP_LEN, RO_EP_RET, RO_OB, RO_AC, RO_ADV_GL, RO_VAL_GL, \
    GRAPH_OUT,\
    GD_CHG, GD_AVG_NUM_UPCLIPS, GD_AVG_NUM_DOWNCLIPS, GD_ACTUAL_CLIPS
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
g_num_timesteps = 200000
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
    # to store change magnitudes at each iteration
    changes = [];

    # to store the average number of upclips (r_t > 1 + epsilon) and downclips
    # (r_t < 1 - epsilon) taken over all batches at each iteration
    avg_num_upclips = [];
    avg_num_downclips = [];
    # - 

    # TODO Delete
#    test_obs = np.random.rand(1, 4) 
#    test_acs = model.eval_policy_var_single(test_obs)
#    test_advs_gl = np.arange(1)
#    test_vals_gl = np.arange(1) + 1
#    model.adam_update(test_obs, test_acs,\
#    test_advs_gl, test_vals_gl)
#    import sys
#    sys.exit()
    # Delete TODO 

    # - training -
    # continue training until timestep limit is reached
    while (timesteps < num_timesteps):
        # - SGD setup - 
        model.update_cpu_networks()

        # get a rollout under this model for training
        rollout = rollout_gen.__next__()
        
        # update old policy function to new policy function
        model.update_old_pol()

        # center advatages
        rollout[RO_ADV_GL] = (rollout[RO_ADV_GL] - rollout[RO_ADV_GL].mean()) 

        # place data into dataset that will shuffle and batch them for training
        data = Dataset({RO_OB:rollout[RO_OB], RO_AC:rollout[RO_AC],\
            RO_ADV_GL:rollout[RO_ADV_GL], RO_VAL_GL:rollout[RO_VAL_GL]})

        # linearly decrease learning rate TODO delete?
#        alpha_decay_factor = \
#                max(1.0 - float(timesteps) / num_timesteps, 0)
        # - 

        # - SGD training -
        # go through all epochs
        for e in range(num_epochs):
            # go through all randomly organized batches
            for batch in data.iterate_once(batch_size):
                # get gradient
                model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL])
        # - 
    
        ratio = model.ratio(rollout[RO_OB], rollout[RO_AC])
        num_upclips = torch.sum(ratio > (1 + model.clip_param)).item()
        num_downclips = torch.sum(ratio < (1 - model.clip_param)).item()
        actual_clips = torch.sum(ratio > (1 + model.clip_param)).item()

        avg_num_upclips.append(num_upclips)
        avg_num_downclips.append(num_downclips)

        # standard deviation after update
        change = model.policy_change()

        # save standard deviation
        changes.append(change)

        # update total timesteps traveled so far
        timesteps += np.sum(rollout[RO_EP_LEN])
        print(avg_num_upclips[-1], avg_num_downclips[-1], changes[-1])
        print("Time Elapsed: {0}; Average Reward: {1}".format(timesteps, 
            np.mean(rollout[RO_EP_LEN][-100:])))
    # - 

    graph_data = {}
    graph_data[GD_CHG] = np.array(changes)
    graph_data[GD_AVG_NUM_UPCLIPS] = np.array(avg_num_upclips)
    graph_data[GD_AVG_NUM_DOWNCLIPS] = np.array(avg_num_downclips)
    graph_data[GD_ACTUAL_CLIPS] = np.array(avg_num_downclips)

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

def graph_chgs_and_clips(chgs, avg_num_upclips, avg_num_downclips, actual_clips):
    iterations = np.arange(chgs.shape[0]) + 1

    plt.figure()
    ax = plt.subplot(211)
    plt.ylabel("count")
    plt.xlabel("Number of Iterations")
    plt.title("Clipping Behavior")

    plt.plot(iterations, actual_clips, linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        #label='$|\{r_t \mid r_t < 1 - \epsilon\}| - |\{r_t \mid r_t > 1 + \epsilon\}|$')
        label='$|\{t \mid (r_t > 1 + \epsilon) \land (G_t > 0)\}|$')
    ax.axhline(color="gray")

    plt.legend()
    
    plt.subplot(212)
    plt.ylabel("$||\\Delta \\pi||$")
    plt.xlabel("Number of Iterations")
    plt.title("Magnitude of Policy Change")
    plt.plot(iterations, chgs, linestyle='-', color=(0.0, 0.0, 0.0),\
            label='$||\\Delta \\pi||$')

    plt.legend()

    plt.tight_layout()

    plt.savefig(GRAPH_OUT)
    plt.show()

def chgs_and_clips_run():
    all_chgs = []
    all_avg_num_upclips = []
    all_avg_num_downclips = []
    all_actual_clips = []
    for seed in range(3):
        graph_data = global_run_seed(seed)
        chgs = graph_data[GD_CHG]
        avg_num_upclips = graph_data[GD_AVG_NUM_UPCLIPS]
        avg_num_downclips = graph_data[GD_AVG_NUM_DOWNCLIPS]
        actual_clips = graph_data[GD_ACTUAL_CLIPS]
        all_chgs.append(chgs)
        all_avg_num_upclips.append(avg_num_upclips)
        all_avg_num_downclips.append(avg_num_downclips)
        all_actual_clips.append(actual_clips)

    chgs_avg = get_average_list_arr(all_chgs)
    avg_num_upclips_avg = get_average_list_arr(all_avg_num_upclips)
    avg_num_downclips_avg = get_average_list_arr(all_avg_num_downclips)
    actual_clips_avg = get_average_list_arr(all_actual_clips)
    
    graph_chgs_and_clips(chgs_avg, avg_num_upclips_avg,\
            avg_num_downclips_avg, actual_clips_avg)

def main():
    chgs_and_clips_run()

if __name__ == '__main__':
    main()
