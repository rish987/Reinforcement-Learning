# File: eps_trainer.py 
# Author(s): Rishikesh Vaishnav
# Created: 08/08/2018
# Experimental program for training two epsilon parameters.
import torch
import torch.optim as optim
import numpy as np
from torch.distributions import normal

target_discrepancy = torch.tensor(-0.3)

learning_rate = 0.01

std = torch.tensor([1.0])
mean_old = torch.tensor([0.0])
mean = torch.tensor([1.0])

dist_old = normal.Normal(mean_old, std)
dist = normal.Normal(mean, std)

eps_down = torch.tensor(0.2, requires_grad=True)
eps_up = torch.tensor(0.2, requires_grad=True)
learnable_params = [eps_down, eps_up]

optimizer = optim.Adam(learnable_params, learning_rate)

for _ in range(100):
    x_up = ((mean ** 2 + mean_old ** 2) + (2 * (std ** 2) * torch.log(1 +\
            eps_up))) / (2 * (mean - mean_old))
            
    x_down = ((mean ** 2 + mean_old ** 2) + (2 * (std ** 2) * torch.log(1 -\
            eps_down))) / (2 * (mean - mean_old))

    int_1 = dist_old.cdf(x_up) - dist_old.cdf(x_down)
    int_2 = dist.cdf(x_up) - dist.cdf(x_down)
    int_3 = 1 - dist_old.cdf(x_up)

    discrepancy = eps_down + ((1 - eps_down) * int_1) - \
                (int_2 + ((eps_up + eps_down) * int_3))

    loss = (discrepancy - target_discrepancy) ** 2

    e_r_down = ((1 - eps_down) * dist_old.cdf(x_down)) + (1 - dist.cdf(x_down))
    e_r_up = ((1 + eps_up) * (1 - dist_old.cdf(x_up))) + dist.cdf(x_up)
    
    print(loss.item())
    print(discrepancy.item())
    print(eps_down.item())
    print(eps_up.item())
    print((e_r_down.item() - 1) + (1 - e_r_up.item()))
    print()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

