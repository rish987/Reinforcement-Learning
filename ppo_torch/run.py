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
    model = PPOModel(env, num_hidden_layers, hidden_layer_size, alpha,\
            clip_param)

    # TODO delete experimenting -

    for dict_name, network in [("value_net", model.value_net), \
        ("pol_net", model.policy_net.policy_net)]:
        # load parameters from tf implementation into model.policy_net
        with open(dict_name + '_state_dict', 'rb') as file:
            state_dict = pickle.load(file)

        for key in state_dict:
            state_dict[key] = torch.tensor(state_dict[key])

        network.load_state_dict(state_dict)

        # get value on dummy observation
        #print(network(torch.tensor([1.0, 2.0, 3.0, 4.0])))

    # - TODO delete experimenting

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

        # TODO delete experimenting - 
        print(model.loss(rollout[RO_OB], rollout[RO_AC],\
                    rollout[RO_ADV_GL], rollout[RO_VAL_GL]))
        import sys
        sys.exit()
        # - TODO delete experimenting

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
            np.mean(rollout[RO_EP_LEN])))

        import sys
        sys.exit()
    # - 

def main():
    # TODO pass in necessary parameters (use no globals)
    train()

if __name__ == '__main__':
    main()
