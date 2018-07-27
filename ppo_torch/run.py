# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Main runner class for implementation of OpenAI PPO, using PyTorch instead of
# TensorFlow.

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
# number of timesteps in a single run (simulated trajectory with fixed
# parameters)
timesteps_per_run = 2048
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
