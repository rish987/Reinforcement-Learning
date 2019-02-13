# File: ppo_model.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/27/2018
# Description:
# Class (PPOModel) that encapsulates relevant PPO parameters and procedures.
from imports import *
from constants import *
from misc_utils import from_numpy_dt, to_numpy_dt, print_message, layer_init

device = torch.device("cuda")

"""
Encapsulates relevant PPO parameters and procedures.
"""
class PPOModel(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size, alpha, \
        clip_param_up, clip_param_down):
        # - set up networks -
        # TODO use lambda here to avoid repeated code
        self.policy_net = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, True,\
            device)

        self.policy_net_cpu = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, False,\
            "cpu")

        self.policy_net_old = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, False,\
            device)

        self.value_net = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1, \
            True).to(device)

        self.value_net_cpu = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1, \
            False).to("cpu")
        # - 

        self.clip_param_up = clip_param_up
        self.clip_param_down = clip_param_down
        
        # set up optimizer
        self.optimizer = optim.Adam(self.trainable_parameters(), alpha)

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
    Executes the policy function stochastically on the given observation 'ob'.
    Runs on CPU-based policy network to avoid wasteful overhead.
    """
    def eval_policy_var_single(self, ob):
        return self.eval_single(self.policy_net_cpu, ob, True)

    """
    Executes the value function on the given observation 'ob'.
    Runs on CPU-based policy network to avoid wasteful overhead.
    """
    def eval_value_single(self, ob):
        return self.eval_single(self.value_net_cpu, ob)

    """
    Executes the specified function on the given observation 'ob'.
    Runs on CPU to avoid wasteful overhead.
    """
    def eval_single(self, network, ob, *args):
        return to_numpy_dt(network(from_numpy_dt(ob, "cpu"), *args))

    """
    Updates the CPU networks with the current parameters.
    """
    def update_cpu_networks(self):
        self.policy_net_cpu.copy_params(self.policy_net)
        self.value_net_cpu.load_state_dict(self.value_net.state_dict())

    """
    Updates the parameters of the model according to the Adam framework.
    """
    def adam_update(self, obs, acs, advs_gl, vals_gl):
        # clear gradients
        self.optimizer.zero_grad()
        pol_loss, val_loss = self.loss(obs, acs, advs_gl, vals_gl)
            
        loss = pol_loss + val_loss
        loss.backward()
        self.optimizer.step()

    """
    Calculates the loss ratio given actions and observations.
    """
    def ratio(self, obs, acs):
        return torch.exp(self.policy_net.logp(acs, obs) - \
            self.policy_net_old.logp(acs, obs))

    """
    Calculates the total loss with the given batch data.
    """
    def loss(self, obs, acs, advs_gl, vals_gl):
        ratio = self.ratio(obs, acs)

        target = from_numpy_dt(advs_gl, device)
        
        # - calculate policy loss - TODO combine into a single operation?
        surr1 = ratio * target
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_param_down, max=1.0 +\
                self.clip_param_up) * target
        pol_loss = -torch.mean(torch.min(surr1, surr2))
        # - 

        # calculate value loss
        val_loss = torch.mean(torch.pow(self.value_net(from_numpy_dt(obs,\
            device)) - from_numpy_dt(vals_gl[:, None], device), 2.0))

        return pol_loss, val_loss

"""
Neural network and standard deviation for the policy function.
"""
class PolicyNetVar(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size, training, \
        device_in):
        self.device = device_in
        self.policy_net = GeneralNet(env, num_hidden_layers, \
            hidden_layer_size, env.action_space.shape[0], \
            training).to(self.device)
        # TODO which requires_grad are necessary? And why zero variance?
        self.logstd = nn.Parameter(torch.zeros(env.action_space.shape[0], \
            requires_grad=False, device=self.device), requires_grad=False)

    """
    Copies the parameters from the specified PolicyNetVar into this one.
    """
    def copy_params(self, other):
        # copy logstd parameter
        self.logstd.data = other.logstd.clone().to(self.device)
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
        # TODO fix
        # yield self.logstd

    def std(self):
        return torch.exp(self.logstd)

    """
    Returns the log-probability of the given action "ac" given the observation
    "ob".
    """
    def logp(self, ac, ob):
        mean = self.policy_net(from_numpy_dt(ob, device))
        ac_t = from_numpy_dt(ac, device)
        dimension = float(ac.shape[1])

        ret = (0.5 * torch.sum(torch.pow((ac_t - mean) / self.std(), 2.0),
            dim=1)) + (0.5 * torch.log(torch.tensor(2.0 * np.pi, device=\
            self.device)) * torch.tensor(dimension, device=self.device))\
            + torch.sum(self.logstd)

        return -ret

"""
Neural network for the policy/value/etc. function.
"""
class GeneralNet(nn.Module):
    def __init__(self, env, num_hidden_layers, hidden_layer_size,\
            output_size, training):
        super(GeneralNet, self).__init__()

        # to hold fully connected layers
        layers = [];

        # shorten syntax for creating a new layer and specially initializing it
        # if necessary
        init_layer_op = lambda m: layer_init(m) if training else m
        new_layer_op = lambda in_size, out_size: \
            init_layer_op(nn.Linear(in_size,out_size))

        # add first layer, coming directly from the observation
        layers.append(new_layer_op(env.observation_space.shape[0],\
            hidden_layer_size))

        # go through all remaining layers
        for _ in range(num_hidden_layers):
            layers.append(new_layer_op(hidden_layer_size, hidden_layer_size))

        # the output layer is an action-space-dimensional value
        layers.append(new_layer_op(hidden_layer_size, output_size))

        self.fc = nn.ModuleList(layers)

        if not training:
            # prevent old policy from being considered in update
            self.fix()

    """
    Disable gradient for the parameters of this module.
    """
    def fix(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        for layer_i in range(len(self.fc) - 1):
            x = F.tanh(self.fc[layer_i](x))
        x = self.fc[-1](x)
        return x
