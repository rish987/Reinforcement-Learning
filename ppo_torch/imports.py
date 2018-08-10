# File: imports.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Necessary imported modules.
import numpy as np
# TODO necessary?
import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import normal
device = torch.device("cuda")
