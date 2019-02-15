# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from constants import *
from misc_utils import set_random_seed, Dataset, from_numpy_dt, \
    graph_data_keys, print_message, clear_out_file, EnvNormalized, EnvToTorch
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
# environment name
g_env_name = "MountainCarContinuous-v0"
# initial epsilon value
g_init_eps = 0.4
# number of timesteps to train over
g_num_timesteps = 300000
# number of timesteps in a single rollout (simulated trajectory with fixed
# parameters)
g_timesteps_per_rollout = 2048
# random seed
g_seed = 1
# -- 

# -- SGD parameters --
# number of training epochs per run
g_num_epochs = 10
# adam learning rate
g_alpha = 3e-6
# number of randomly selected timesteps to use in a single parameter update
g_batch_size = 64
# --

# -- GAE parameters --
# gamma and lambda factors used in Generalized Advantage Estimation (Schulman
# et. al.) to trade off between variance and bias in return/advantage
# approximation
g_gamma = 0.99
g_lambda_ = 0.95
# -- 
# - 

"""
Trains a PPO agent according to given parameters and reports results.
"""
def train(hidden_layer_size, num_hidden_layers, num_timesteps, \
    timesteps_per_rollout, seed, clip_param_up, clip_param_down, num_epochs, \
    alpha, batch_size, gamma, lambda_, env_name):

    # TODO make into a file constant (prefixed with "g_")?
    device = torch.device("cuda")

    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env = EnvNormalized(env)
    env = EnvToTorch(env, device)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    model = PPOModel(env, num_hidden_layers, hidden_layer_size, alpha,\
            clip_param_up, clip_param_down, device)

    # total number of timesteps trained so far
    timesteps = 0

    # generator for getting rollouts
    rollout_gen = get_rollout(env, model, timesteps_per_rollout, gamma, lambda_)
    # - 

    # - training -
    # continue training until timestep limit is reached
    while (timesteps < num_timesteps):
        # - SGD setup - 
        # get a rollout under this model for training
        rollout = rollout_gen.__next__()
        
        # center advatages
        rollout[RO_ADV_GL] = (rollout[RO_ADV_GL] - rollout[RO_ADV_GL].mean()) 

        # place data into dataset that will shuffle and batch them for training
        data = Dataset({RO_OB:rollout[RO_OB], RO_AC:rollout[RO_AC],\
            RO_ADV_GL:rollout[RO_ADV_GL], RO_VAL_GL:rollout[RO_VAL_GL]})

        # - SGD training -
        # go through all epochs
        for e in range(num_epochs):
            # go through all randomly organized batches
            for batch in data.iterate_once(batch_size):
                # get gradient
                model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL])
                first = False
        # - 
    
        # - gather graph data -
        avg_ret = np.mean(rollout[RO_EP_RET][-100:])
        # - 

        status = "Average Reward: {}".format(avg_ret)
        print_message(status)
    # - 

    return graph_data

def main():
    clear_out_file()
    return train(hidden_layer_size=g_hidden_layer_size, \
        num_hidden_layers=g_num_hidden_layers, num_timesteps=g_num_timesteps, \
        timesteps_per_rollout=g_timesteps_per_rollout, seed=g_seed, \
        clip_param_up=g_init_eps, clip_param_down=g_init_eps, \
        num_epochs=g_num_epochs, alpha=g_alpha, \
        batch_size=g_batch_size, gamma=g_gamma, lambda_=g_lambda_, \
        env_name=g_env_name)

if __name__ == '__main__':
    main()
