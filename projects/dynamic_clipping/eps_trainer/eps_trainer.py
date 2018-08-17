# File: eps_trainer.py 
# Author(s): Rishikesh Vaishnav
# Created: 08/08/2018
# Experimental program for training two epsilon parameters.
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import normal
def get_x_lim_mult(eps_mult, mean, mean_old):
    return ((mean ** 2 - mean_old ** 2) + (2 * (std ** 2) * torch.log(1 +\
            eps_mult))) / (2 * (mean - mean_old))

def get_x_lim(eps, up, mean, mean_old):
    mult = 1 if up else -1
    return get_x_lim_mult(mult * eps, mean, mean_old)

def get_discrepancy_and_penalty_contributions(eps_down, eps_up, dist, dist_old):
    x_up = get_x_lim(eps_up, True, dist.mean, dist_old.mean)
    x_down = get_x_lim(eps_down, False, dist.mean, dist_old.mean)
    #print((eps_up).item())
    #print((torch.exp(dist.log_prob(x_up) - dist_old.log_prob(x_up))).item())

    int_1 = dist_old.get_prob_between(x_down, x_up)
    int_2 = dist.get_prob_between(x_down, x_up)
    int_3 = dist_old.get_prob_above(x_up)
    int_4 = dist.get_prob_above(x_up)
    #int_1 = dist_old.get_prob_between(x_up, x_down)
    #int_2 = dist.get_prob_between(x_up, x_down)
    #int_3 = dist_old.get_prob_below(x_up)
    #int_4 = dist.get_prob_below(x_up)
#    print(dist.get_prob_above(x_up) + (1 + eps_up) *\
#            dist_old.get_prob_below(x_up))
#    print(dist.get_prob_below(x_down) + (1 - eps_down) *\
#            dist_old.get_prob_above(x_down))
#    print(-eps_down - ((1 - eps_down) * (dist_old.get_prob_between(x_up,\
#        x_down) + dist_old.get_prob_below(x_up))) \
#        + dist.get_prob_between(x_up, x_down) + dist.get_prob_below(x_up))

    discrepancy = eps_down + ((1 - eps_down) * int_1) - \
                (int_2 + ((eps_up + eps_down) * int_3))

    penalty_contributions = \
        (2 * int_4) - ((2 + eps_up - eps_down)* int_3) - eps_down - \
        ((1 - eps_down) * int_1) + int_2

    return discrepancy, penalty_contributions

def get_discrepancy_and_penalty_contribution_losses(eps_down, eps_up, dist,\
        dist_old, target_discrepancy, target_penalty_contribution):
    discrepancy, penalty_contributions =\
        get_discrepancy_and_penalty_contributions(eps_down, eps_up, dist,\
                dist_old)

    discrepancy_loss = (discrepancy - target_discrepancy) ** 2
    penalty_contribution_loss = (penalty_contributions -\
            target_penalty_contribution) ** 2
    return discrepancy_loss, penalty_contribution_loss, \
        discrepancy, penalty_contributions

class NormalUtil(normal.Normal):
    def get_prob_between(self, down, up):
        return self.cdf(up) - self.cdf(down)

    def get_prob_above(self, val):
        return 1 - self.cdf(val)

    def get_prob_below(self, val):
        return self.cdf(val)


std = torch.tensor([1.0])
mean_old_orig = torch.tensor([0.0])
mean_orig = torch.tensor([0.1408])

g_mean_old = torch.tensor([0.0])
g_mean = torch.abs(mean_orig - mean_old_orig)

g_dist_old = NormalUtil(g_mean_old, std)
g_dist = NormalUtil(g_mean, std)

g_eps_down = torch.tensor(0.2, requires_grad=True)
g_eps_up = torch.tensor(0.2, requires_grad=True)
learnable_params = [g_eps_down, g_eps_up]

learning_rate = 0.01
optimizer = optim.Adam(learnable_params, learning_rate)

discrepancy, penalty_contributions = \
    get_discrepancy_and_penalty_contributions(g_eps_down, g_eps_up, g_dist, \
    g_dist_old) 

g_target_discrepancy = torch.tensor(0.02)
g_target_penalty_contribution = torch.tensor(penalty_contributions.item())

for _ in range(20):
    discrepancy_loss, penalty_contribution_loss, \
        discrepancy, penalty_contributions = \
        get_discrepancy_and_penalty_contribution_losses(g_eps_down, g_eps_up, \
        g_dist, g_dist_old, g_target_discrepancy, g_target_penalty_contribution)

    loss = discrepancy_loss + penalty_contribution_loss
    
    print("Loss: {0}".format(loss.item()))
    print("Discrepancy: {0}".format(discrepancy.item()))
    print("Low epsilon: {0}".format(g_eps_down.item()))
    print("High epsilon: {0}".format(g_eps_up.item()))
    print("Total penalty contributions: {0}".format(\
        penalty_contributions.item()))
    print()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
