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
    def __init__(self, env, num_hidden_layers, hidden_layer_size, alpha, \
        clip_param):
        # - set up networks -
        self.policy_net = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size)

        self.policy_net_old = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size)

        # prevent old policy from being considered in update
        for param in self.policy_net_old.parameters():
            param.requires_grad = False

        self.value_net = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1)
        # - 

        self.clip_param = clip_param

        # set up optimizer
        self.optimizer = optim.SGD(self.trainable_parameters(), 0.001)

    """
    Sets the old policy to have the same parameters as the new policy.
    """
    def update_old_pol(self):
        self.policy_net_old.copy_params(self.policy_net)

    """
    Generator for all parameters to be trained by this model.
    """
    def trainable_parameters(self):
        for parameter in self.policy_net.parameters():
            yield parameter
        for parameter in self.value_net.parameters():
            yield parameter

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
    Updates the parameters of the model according to the Adam framework.
    """
    def adam_update(self, obs, acs, advs_gl, vals_gl):
        # clear gradients
        self.optimizer.zero_grad()
        pol_loss, val_loss = self.loss(obs, acs, advs_gl, vals_gl)
        loss = pol_loss + val_loss
        loss.backward()
        # TODO is adam going in the right direction?
        self.optimizer.step()
        
    """
    Calculates the total loss with the given batch data.
    """
    def loss(self, obs, acs, advs_gl, vals_gl):
        ratio = torch.exp(self.policy_net.logp(acs, obs) - \
                self.policy_net_old.logp(acs, obs))

        # - calculate policy loss -
        surr1 = ratio * torch.from_numpy(advs_gl)
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_param, max=1.0 +\
                self.clip_param) \
            * torch.from_numpy(advs_gl)
        pol_loss = -torch.mean(torch.min(surr1, surr2))
        # - 

        # calculate value loss
        val_loss = torch.mean(torch.pow(self.value_net(torch.from_numpy(obs)) \
                - torch.from_numpy(vals_gl[:, None]), 2.0))
        #print("pol_loss: {0} \t val_loss: {1}".format(pol_loss, val_loss))

        return pol_loss, val_loss

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
    Copies the parameters from the specified PolicyNetVar into this one.
    """
    def copy_params(self, other):
        # copy logstd parameter
        self.logstd.data = other.logstd.clone()
        # copy all other parameters
        self.policy_net.load_state_dict(other.policy_net.state_dict())

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
                std=self.std())
    """
    Returns all of this policy's parameters.
    """
    def parameters(self):
        for param in self.policy_net.parameters():
            yield param
        yield self.logstd

    def std(self):
        return torch.exp(self.logstd)

    """
    Returns the log-probability of the given action "ac" given the observation
    "ob".
    """
    def logp(self, ac, ob):
        mean = self.policy_net(torch.from_numpy(ob))
        ac_t = torch.from_numpy(ac)
        dimension = float(ac.shape[1])

        ret = (0.5 * torch.sum(torch.pow((ac_t - mean) / self.std(), 2.0),
            dim=1)) + (0.5 * torch.log(torch.tensor(2.0 * np.pi)) *\
            torch.tensor(dimension)) + torch.sum(self.logstd)

        return -ret
"""
Neural network for the policy/value/etc. function.
"""
class GeneralNet(nn.Module):
    def __init__(self, env, num_hidden_layers, hidden_layer_size,\
            last_layer_size):
        super(GeneralNet, self).__init__()

        # to hold fully connected layers
        layers = [];

        # TODO use correct initializers?
        
        # add first layer, coming directly from the observation
        layers.append(nn.Linear(env.observation_space.shape[0],\
            hidden_layer_size))

        # go through all remaining layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))

        # the output layer is an action-space-dimensional value
        layers.append(nn.Linear(hidden_layer_size, \
            last_layer_size))

        self.fc = nn.ModuleList(layers)

    def forward(self, x):
        for layer_i in range(len(self.fc) - 1):
            x = F.tanh(self.fc[layer_i](x))
        x = self.fc[-1](x)
        return x
