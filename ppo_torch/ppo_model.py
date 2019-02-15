# File: ppo_model.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/27/2018
# Description:
# Class (PPOModel) that encapsulates relevant PPO parameters and procedures.
from imports import *
from constants import *
from misc_utils import from_numpy_dt, to_numpy_dt, print_message, layer_init

"""
Encapsulates relevant PPO parameters and procedures.
"""
class PPOModel(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size, alpha, \
        clip_param_up, clip_param_down, device):

        self.device = device

        # - set up networks -
        # get training models
        self.policy = Policy(env, num_hidden_layers, hidden_layer_size, True,\
                self.device)
        self.value_net = self.policy.policy_net.value_net

        # get dummy models
        self.policy_cpu = Policy(device_in="cpu")
        self.policy_old = Policy(device_in=self.device)
        self.value_net_cpu = GeneralNet(device_in="cpu")
        # - 

        self.clip_param_up = clip_param_up
        self.clip_param_down = clip_param_down
        
        # set up optimizer
        self.optimizer = optim.Adam(self.trainable_parameters(), alpha)

    """
    Sets the old policy to have the same parameters as the new policy.
    """
    def update_old_pol(self):
        self.policy_old.copy_params_fix(self.policy)

    """
    Generator for all parameters to be trained by this model.
    """
    def trainable_parameters(self):
        for parameter in self.policy.parameters():
            yield parameter
        for parameter in self.value_net.parameters():
            yield parameter

    """
    Executes the policy function stochastically on the given observation 'ob'.
    Runs on CPU-based policy network to avoid wasteful overhead.
    """
    def eval_policy_var_single(self, ob):
        return self.eval_single(self.policy_cpu, ob, True)

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
        self.policy_cpu.copy_params_fix(self.policy)
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
        return torch.exp(self.policy.logp(acs, obs) - \
            self.policy_old.logp(acs, obs))

    """
    Calculates the total loss with the given batch data.
    """
    def loss(self, obs, acs, advs_gl, vals_gl):
        ratio = self.ratio(obs, acs)

        target = from_numpy_dt(advs_gl, self.device)
        
        # - calculate policy loss - TODO combine into a single operation?
        surr1 = ratio * target
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_param_down, max=1.0 +\
                self.clip_param_up) * target
        pol_loss = -torch.mean(torch.min(surr1, surr2))
        # - 

        # calculate value loss
        val_loss = torch.mean(torch.pow(self.value_net(from_numpy_dt(obs,\
            self.device)) - from_numpy_dt(vals_gl[:, None], self.device), 2.0))

        return pol_loss, val_loss

"""
Neural network and standard deviation for the policy function.
"""
class Policy(object):
    def __init__(self, env=None, num_hidden_layers=None,\
        hidden_layer_size=None, training=False, device_in=None):
        self.device = device_in
        # randomly initalize networks
        if training:
            self.policy_net = GeneralNet(env, num_hidden_layers, \
                hidden_layer_size, env.action_space.shape[0], \
                True, True, self.device)
            self.logstd = nn.Parameter(torch.zeros(env.action_space.shape[0], \
                device=self.device))
            self.logstd.requires_grad = True
        # create dummy networks to copy something else into
        else:
            self.policy_net = GeneralNet(device_in=self.device)
            self.logstd = nn.Parameter()
            # TODO necessary?
            self.fix()

    """
    Copies the parameters from the specified Policy into this one, and labels
    this Policy's parameters as fixed.
    """
    def copy_params_fix(self, other):
        # copy logstd parameter
        self.logstd.data = other.logstd.clone().to(self.device)
        # copy all other parameters
        self.policy_net.load_state_dict(other.policy_net.state_dict())
        # TODO necessary?
        self.fix()

    """
    Disable gradient for the parameters of this Policy.
    """
    def fix(self):
        self.policy_net.fix();
        self.logstd.requires_grad = False

    """
    Returns an action, stochastically via gaussian or deterministically, given
    the output from self.policy_net and the log standard deviation self.logstd.  
    """
    def __call__(self, ob, stochastic):
        # output mean
        if not stochastic:
            return self.policy_net(ob)
        # output mean with some gaussian noise according to self.logstd
        else:
            return torch.distribitions.Normal(self.policy_net(ob),\
                self.std())
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
        mean = self.policy_net(from_numpy_dt(ob, self.device))
        ac_t = from_numpy_dt(ac, self.device)
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
    def __init__(self, env=None, num_hidden_layers=None,\
            hidden_layer_size=None, output_size=None, training=False,\
            create_value_net=False, device_in=None):
        super(GeneralNet, self).__init__()

        # this is not a dummy network
        if training:
            # to hold fully connected layers
            layers = [];

            # shorten syntax for creating a new layer and specially
            # initializing it if necessary
            init_layer_op = lambda m, gain: layer_init(m, gain) if training \
                else m
            new_layer_op = lambda in_size, out_size, gain: \
                init_layer_op(nn.Linear(in_size, out_size), gain)

            # add first layer, coming directly from the observation
            layers.append(new_layer_op(env.observation_space.shape[0],\
                hidden_layer_size, np.sqrt(2)))

            # go through all remaining layers
            for _ in range(num_hidden_layers - 1):
                layers.append(new_layer_op(hidden_layer_size, \
                    hidden_layer_size, np.sqrt(2)))

            # option to construct value net here for purpose of
            # initialization value alignment with other code
            if (create_value_net):
                self.value_net = \
                    GeneralNet(env, num_hidden_layers, hidden_layer_size, 1, \
                    True, False, device_in)

            # the output layer is an action-space-dimensional value
            layers.append(new_layer_op(hidden_layer_size, output_size,
                1 if create_value_net else np.sqrt(2)))

            self.fc = nn.ModuleList(layers)

        self.to(device_in)

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
