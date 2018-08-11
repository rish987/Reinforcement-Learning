# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed, Dataset, from_numpy_dt
from misc_utils import \
    RO_EP_LEN, RO_EP_RET, RO_OB, RO_AC, RO_ADV_GL, RO_VAL_GL, \
    GRAPH_OUT,\
    GD_CHG, GD_AVG_NUM_UPCLIPS, GD_AVG_NUM_DOWNCLIPS, GD_AVG_UPCLIP_DED,\
    GD_AVG_DOWNCLIP_DED, GD_EP_RETS, GD_TIMESTEPS,\
    graph_data_keys
from ppo_model import PPOModel
from rollout import get_rollout

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
g_clip_param_up = 0.1
g_clip_param_down = 0.1

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
g_env_name = "Swimmer-v2"

"""
Trains a PPO agent according to given parameters and reports results.
"""
def train(hidden_layer_size, num_hidden_layers, num_timesteps, \
    timesteps_per_rollout, seed, clip_param_up, clip_param_down, num_epochs, \
    alpha, batch_size, gamma, lambda_, env_name, experimental=False):
    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env.seed(seed)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    model = PPOModel(env, num_hidden_layers, hidden_layer_size, alpha,\
            clip_param_up, clip_param_down)

    # total number of timesteps trained so far
    timesteps = 0

    # generator for getting rollouts
    rollout_gen = get_rollout(env, model, timesteps_per_rollout, gamma, lambda_)
    # - 

    # - initialize graph data -
    graph_data = {}
    for key in graph_data_keys:
        graph_data[key] = []
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

        # center advatages
        rollout[RO_ADV_GL] = (rollout[RO_ADV_GL] - rollout[RO_ADV_GL].mean()) 

        # place data into dataset that will shuffle and batch them for training
        data = Dataset({RO_OB:rollout[RO_OB], RO_AC:rollout[RO_AC],\
            RO_ADV_GL:rollout[RO_ADV_GL], RO_VAL_GL:rollout[RO_VAL_GL]})

        # linearly decrease learning rate TODO delete?
#        alpha_decay_factor = \
#                max(1.0 - float(timesteps) / num_timesteps, 0)
        # - 

        # have we just started training?
        first_train = True

        # - SGD training -
        # go through all epochs
        for e in range(num_epochs):
            # go through all randomly organized batches
            for batch in data.iterate_once(batch_size):
                # get gradient
                model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL], experimental, \
                    first_train)
                if first_train:
                    first_train = False
        # - 
    
        # - gather graph data -
        ratio = model.ratio(rollout[RO_OB], rollout[RO_AC])
        #clip_param_up_opt, clip_param_down_opt = \
        #    model.optimize_clip_params(rollout[RO_OB])
        clip_param_up_tensor = torch.tensor(model.clip_param_up, device=device)
        clip_param_down_tensor = \
            torch.tensor(model.clip_param_down, device=device)
        avg_ratio_upclipped = torch.min(ratio, \
            1 + clip_param_up_tensor).mean().item()
        avg_ratio_downclipped = torch.max(ratio, \
            1 - clip_param_down_tensor).mean().item()
        avg_ratio = ratio.mean().item()
        avg_upclip_ded = avg_ratio - avg_ratio_upclipped
        avg_downclip_ded = avg_ratio_downclipped - avg_ratio
        num_upclips = torch.sum(ratio > (1 + clip_param_up_tensor)).item()
        num_downclips = torch.sum(ratio < (1 - clip_param_down_tensor)).item()
        actual_clips = torch.sum(ratio > (1 + clip_param_up_tensor)).item()
        avg_ret = np.mean(rollout[RO_EP_RET][-100:])

        # update total timesteps traveled so far
        timesteps += np.sum(rollout[RO_EP_LEN])

        graph_data[GD_EP_RETS].append(avg_ret)
        graph_data[GD_TIMESTEPS].append(timesteps)
        graph_data[GD_AVG_NUM_UPCLIPS].append(num_upclips)
        graph_data[GD_AVG_NUM_DOWNCLIPS].append(num_downclips)
        graph_data[GD_AVG_UPCLIP_DED].append(avg_upclip_ded)
        graph_data[GD_AVG_DOWNCLIP_DED].append(avg_downclip_ded)

        # standard deviation after update
        change = model.policy_change()

        # save change magnitude,
        graph_data[GD_CHG].append(change)
        # - 

        #print(avg_num_upclips[-1], avg_num_downclips[-1], changes[-1])
        print("Time Elapsed: {0}; Average Reward: {1}".format(timesteps, 
            avg_ret))
    # - 

    return graph_data

def global_run_seed(seed, experimental=False, env_name=g_env_name):
    return train(hidden_layer_size=g_hidden_layer_size, \
        num_hidden_layers=g_num_hidden_layers, num_timesteps=g_num_timesteps, \
        timesteps_per_rollout=g_timesteps_per_rollout, seed=seed, \
        clip_param_up=g_clip_param_up, clip_param_down=g_clip_param_down, \
        num_epochs=g_num_epochs, alpha=g_alpha, \
        batch_size=g_batch_size, gamma=g_gamma, lambda_=g_lambda_, \
        env_name=env_name, experimental=experimental)

def get_average_list_arr(list_arr):
    arr_len = min([len(arr) for arr in list_arr]);
    list_arr_cropped = [arr[:arr_len] for arr in list_arr]
    return np.average(np.array(list_arr_cropped), axis=0)

def get_average_data(all_data):
    comb_all_data = {}
    for key in all_data[0]:
        comb_all_data[key] = []

    for data in all_data:
        for key in data:
            comb_all_data[key].append(data[key])

    avg_all_data = {}
    for key in comb_all_data:
        avg_all_data[key] = get_average_list_arr(comb_all_data[key])

    return avg_all_data
        
    
def data_run(experimental=False, env_name=g_env_name):
    all_data = []
    for seed in range(3):
        print("Run {0}:".format(seed + 1))
        graph_data = global_run_seed(seed, experimental=experimental, \
            env_name=env_name)
        all_data.append(graph_data)
    print()

    return get_average_data(all_data)

environments_all = ['InvertedPendulum-v2', 'Reacher-v2',\
    'InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2',\
    'Swimmer-v2', 'Walker2d-v2']

environments_sub = ['InvertedPendulum-v2',\
    'InvertedDoublePendulum-v2', 'Hopper-v2',\
    'Swimmer-v2', 'Walker2d-v2']

def main():
    env_name = g_env_name
    for env_name in environments_sub:
        print("Environment {0}".format(env_name))
        print("Running control...")
        data_contr = data_run(experimental=False, env_name=env_name)
        print("Running experimental...")
        data_exp = data_run(experimental=True, env_name=env_name)
        with open("data_contr_{0}.dat".format(env_name), 'wb') as file:
            pickle.dump(data_contr, file)
        with open("data_exp_{0}.dat".format(env_name), 'wb') as file:
            pickle.dump(data_exp, file)

if __name__ == '__main__':
    main()
