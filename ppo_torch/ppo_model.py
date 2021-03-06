# File: ppo_model.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/27/2018
# Description:
# Class (PPOModel) that encapsulates relevant PPO parameters and procedures.
from imports import *
from misc_utils import from_numpy_dt, to_numpy_dt, NormalUtil, \
    get_discrepancy_and_penalty_contributions, \
    get_discrepancy_and_penalty_contribution_losses, print_message

"""
Encapsulates relevant PPO parameters and procedures.
"""
class PPOModel(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size, alpha, \
        clip_param_up, clip_param_down):
        # - set up networks -
        self.policy_net = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, False,
            device)

        self.policy_net_cpu = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, False,\
            "cpu")

        self.policy_net_old = \
            PolicyNetVar(env, num_hidden_layers, hidden_layer_size, True,\
            device)

        self.value_net = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1).to(device)

        self.value_net_cpu = \
            GeneralNet(env, num_hidden_layers, hidden_layer_size, 1).to("cpu")
        # - 

        self.clip_param_up = clip_param_up
        self.clip_param_down = clip_param_down
        
        # - set up clip parameter optimizer - 
        self.eps_up = torch.tensor(self.clip_param_up, requires_grad=True,\
                device=device)
        self.eps_down = torch.tensor(self.clip_param_down, requires_grad=True,\
                device=device)
        learnable_params = [self.eps_down, self.eps_up]
        learning_rate = 0.0001
        self.clip_param_optimizer = optim.Adam(learnable_params, learning_rate)
        # - 


        # set up optimizer
        self.optimizer = optim.Adam(self.trainable_parameters(), alpha)

        std = self.policy_net.std()[0]

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
    Gets the magnitude of the differences between the old and new policies.
    """
    def policy_change(self):
        total_mag_sq = 0
        for param1, param2 in zip(self.policy_net.parameters(), \
            self.policy_net_old.parameters()):
            diff = param1 - param2
            diffmag = torch.sum(diff ** 2).item()
            total_mag_sq += diffmag

        return total_mag_sq

    """
    Returns clip params, adjusted to minimize expected penalty contribution
    discrepancy between positive and negative estimators, while maintaining the
    same total expected penalty contributions. Uses 0 as the original mean and
    the absolute average distance from the mean as the the new mean.
    """
    def optimize_clip_params(self, obs):
        obs = from_numpy_dt(obs)
        # TODO globalize std parameter
        std = self.policy_net.std()[0]

        # - set up distributions - 
        mean = torch.mean(torch.abs(self.policy_net(obs, False) -\
            self.policy_net_old(obs, False))).detach().view(1)

        dist = NormalUtil(mean, std)
        # - 

        # get initial values
        discrepancy, penalty_contributions = \
            get_discrepancy_and_penalty_contributions(self.eps_down, \
            self.eps_up, dist, std) 

        # set targets
        target_discrepancy = torch.tensor(0.0, device=device)
        target_penalty_contribution = \
            torch.tensor(penalty_contributions.item(), device=device)

        loss_threshold = 1e-8
        max_iters = 10000

        num_iters = 0

        losses = np.array([loss_threshold + 1] * 1)

        while losses.mean() > loss_threshold:
        #for _ in range(20):
            num_iters += 1
            if num_iters > max_iters:
                print_message("WARNING: reached iteration limit. Cutting "
                    "evaluation short with loss={0}".format(losses.mean()))
                break;
            discrepancy_loss, penalty_contribution_loss, \
                discrepancy, penalty_contributions = \
                get_discrepancy_and_penalty_contribution_losses(self.eps_down,\
                self.eps_up, dist, target_discrepancy,\
                target_penalty_contribution, std)

            loss = discrepancy_loss + penalty_contribution_loss
            
            #print("Loss: {0}".format(loss.item()))
            #print("Discrepancy: {0}".format(discrepancy.item()))
            #print("Low epsilon: {0}".format(eps_down.item()))
            #print("High epsilon: {0}".format(eps_up.item()))
            #print("Total penalty contributions: {0}".format(\
            #    penalty_contributions.item()))
            #print()
            assert (not math.isnan(self.eps_down.item())) and \
                    (not math.isnan(self.eps_up.item()))

            self.clip_param_optimizer.zero_grad()
            loss.backward()
            self.clip_param_optimizer.step()

            losses[0:-1] = losses[1:]
            losses[-1] = loss.item()

        print(num_iters)

        self.clip_param_up = self.eps_up.item()
        self.clip_param_down = self.eps_down.item()

    def decay_clip_param_learning_rate(self, rate):
        for g in self.clip_param_optimizer.param_groups:
            g['lr'] *= rate
        
    """
    Calculates the total loss with the given batch data.
    """
    def loss(self, obs, acs, advs_gl, vals_gl):
        ratio = self.ratio(obs, acs)

        target = from_numpy_dt(advs_gl)
        #print(target.mean().item())
        
        # - calculate policy loss -
        surr1 = ratio * target
        surr2 = torch.clamp(ratio, min=1.0 - self.clip_param_down, max=1.0 +\
                self.clip_param_up) * target
        pol_loss = -torch.mean(torch.min(surr1, surr2))
        # - 

        # calculate value loss
        val_loss = torch.mean(torch.pow(self.value_net(from_numpy_dt(obs)) \
                - from_numpy_dt(vals_gl[:, None]), 2.0))
        #print("pol_loss: {0} \t val_loss: {1}".format(pol_loss, val_loss))

        return pol_loss, val_loss

"""
Neural network and standard deviation for the policy function.
"""
class PolicyNetVar(object):
    def __init__(self, env, num_hidden_layers, hidden_layer_size, fixed, \
        device_in):
        self.device = device_in
        self.policy_net = GeneralNet(env, num_hidden_layers, \
            hidden_layer_size, env.action_space.shape[0]).to(self.device)
        self.logstd = nn.Parameter(torch.zeros(env.action_space.shape[0], \
            requires_grad=False, device=self.device), requires_grad=False)

        if fixed:
            # prevent old policy from being considered in update
            for param in self.policy_net.parameters():
                param.requires_grad = False


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
        mean = self.policy_net(from_numpy_dt(ob))
        ac_t = from_numpy_dt(ac)
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
