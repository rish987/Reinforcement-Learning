# File: ppo_model.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/27/2018
# Description:
# Class (PPOModel) that encapsulates relevant PPO parameters and procedures.
from imports import *

"""
Encapsulates relevant PPO parameters and procedures.
"""
class PPOModel(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size):
        self.policy_net = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size)
        self.policy_net_old = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size)

        self.value_net = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1)

    """
    Executes the policy function on the given observation 'ob'.
    """
    def eval_policy(self, ob):
        return self.policy_net(ob, False)

    """
    Executes the policy function stochastically on the given observation 'ob'.
    """
    def eval_policy_var(self, ob):
        return self.policy_net(ob, True)

    """
    Executes the value function on the given observation 'ob'.
    """
    def eval_value(self, ob):
        return self.value_net(ob)

"""
Neural network and standard deviation for the policy function.
"""
class PolicyNetVar(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size):
        self.policy_net = GeneralNet(env, num_hidden_layers, \
            hidden_layer_size, env.action_space.shape[0])
        self.logstd = torch.zeros(env.action_space.shape[0], \
            requires_grad=True)

    """
    Returns an action, stochastically or deterministically, given the output
    from self.policy_net and the gaussian log standard distribution
    self.logstd.  
    """
    def __call__(self, ob, stochastic):
        # output mean
        if not stochastic:
            return self.policy_net(ob)
        # output mean with some gaussian noise according to self.logstd
        else:
            return torch.normal(mean=self.policy_net(ob),\
                std=torch.exp(self.logstd))
"""
Neural network for the policy/value/etc. function.
"""
class GeneralNet(nn.Module):
    def __init__(self, env, num_hidden_layers, hidden_layer_size,\
            last_layer_size):
        super(GeneralNet, self).__init__()

        # to hold fully connected layers
        self.fc = [];

        # TODO use correct initializers?
        
        # add first layer, coming directly from the observation
        self.fc.append(nn.Linear(env.observation_space.shape[0],\
            hidden_layer_size))

        # go through all remaining layers
        for _ in range(num_hidden_layers - 1):
            self.fc.append(nn.Linear(hidden_layer_size, hidden_layer_size))

        # the output layer is an action-space-dimensional value
        self.fc.append(nn.Linear(hidden_layer_size, \
            last_layer_size))

    def forward(self, x):
        for layer_i in range(len(self.fc) - 1):
            x = F.tanh(self.fc[layer_i](x))
        x = self.fc[-1](x)
        return x
