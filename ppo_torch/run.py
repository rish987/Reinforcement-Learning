# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed, Dataset
from misc_utils import RO_EP_LEN, RO_EP_RET, RO_OB, RO_AC, RO_ADV_GL, RO_VAL_GL
from ppo_model import PPOModel
from rollout import get_rollout
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

# - hyperparameters -
# -- neural network parameters --
# number of nodes in each hidden layer
g_hidden_layer_size = 64
# number of hidden layers
g_num_hidden_layers = 2
# --

# -- run parameters --
# number of timesteps to train over
g_num_timesteps = 1e6
# number of timesteps in a single rollout (simulated trajectory with fixed
# parameters)
g_timesteps_per_rollout = 2048
# random seed
g_seed = 0
# -- 

# epsilon as described by Schulman et. al.
g_clip_param = 0.2

# -- SGD parameters --
# number of training epochs per run
g_num_epochs = 10
# adam learning rate
g_alpha = 3e-4
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

# TODO replace with passed-in environment
g_env_name = "InvertedPendulum-v2"

"""
Trains a PPO agent according to given parameters and reports results.
"""
def train(hidden_layer_size, num_hidden_layers, num_timesteps, \
    timesteps_per_rollout, seed, clip_param, num_epochs, alpha, \
    batch_size, gamma, lambda_, env_name, device):
    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env.seed(seed)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    model = PPOModel(env, num_hidden_layers, hidden_layer_size, alpha,\
            clip_param, device)

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

        # update old policy function to new policy function
        model.update_old_pol()

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

        # update total timesteps traveled so far
        timesteps += np.sum(rollout[RO_EP_LEN])
        print("Time Elapsed: {0}; Average Reward: {1}".format(timesteps, 
            np.mean(rollout[RO_EP_LEN][-100:])))
    # - 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(hidden_layer_size=g_hidden_layer_size, \
        num_hidden_layers=g_num_hidden_layers, num_timesteps=g_num_timesteps, \
        timesteps_per_rollout=g_timesteps_per_rollout, seed=g_seed, \
        clip_param=g_clip_param, num_epochs=g_num_epochs, alpha=g_alpha, \
        batch_size=g_batch_size, gamma=g_gamma, lambda_=g_lambda_, \
        env_name=g_env_name, device=device)

if __name__ == '__main__':
    main()
