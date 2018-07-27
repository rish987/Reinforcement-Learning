# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner file for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed
from ppo_model import PPOModel
from rollout import get_rollout
# TODO too slow to use torch.DoubleTensor instead of torch.FloatTensor?
torch.set_default_tensor_type('torch.DoubleTensor')

# - hyperparameters -
# -- neural network parameters --
# number of nodes in each hidden layer
hidden_layer_size  =  64
# number of hidden layers
num_hidden_layers  =  2
# --

# -- run parameters --
# number of timesteps to train over
num_timesteps  =  1e6
# number of timesteps in a single rollout (simulated trajectory with fixed
# parameters)
timesteps_per_rollout = 2048
# random seed
seed = 0
# -- 

# epsilon as described by Schulman et. al.
clip_param = 0.2

# -- SGD parameters --
# number of training epochs per run
num_epochs = 10
# adam learning rate
alpha = 3e-4
# number of randomly selected timesteps to use in a single parameter update
batch_size = 64
# --

# -- GAE parameters --
# gamma and lambda factors used in Generalized Advantage Estimation (Schulman
# et. al.) to trade off between variance and bias in return/advantage
# approximation
gamma = 0.99
lambda_ = 0.95
# -- 
# - 

# TODO replace with passed-in environment
env_name = "InvertedPendulum-v2"

"""
Trains a PPO agent according to given parameters and reports results.
"""
def train():
    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env.seed(seed)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    # TODO implement PPOModel ctor, pass in relevant parameters
    model = PPOModel(env, num_hidden_layers, hidden_layer_size)

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
        import sys
        sys.exit()

        # update old policy function to new policy function
        # TODO implement PPOModel.update_old_pol()
        model.update_old_pol()

        # place data into dataset that will shuffle and batch them for training
        # TODO implement Dataset ctor
        data = Dataset(dict(ob=rollout[RO_OB], ac=rollout[RO_AC],\
            adv=rollout[RO_ADV_GL], val=rollout[RO_VAL_GL]))

        # linearly decrease learning rate
        alpha_decay_factor = \
                max(1.0 - float(timesteps) / num_timesteps, 0)
        # - 

        # - SGD training -
        # go through all epochs
        for _ in num_epochs:
            # go through all randomly organized batches
            # TODO implement Dataset.iterate()
            for batch in d.iterate(batch_size):
                # get gradient
                # TODO implement PPOModel.adam_update()
                model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL])
        # - 

        # update total timesteps traveled so far
        timesteps += rollout[RO_EP_LEN].sum()
    # - 

def main():
    # TODO pass in necessary parameters (use no globals)
    train()

if __name__ == '__main__':
    main()
