# File: misc_utils.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Miscellaneous utility functions.
from imports import *
# - graphing parameters -
# default graph output file
GRAPH_OUT = "graph_out.pgf"

# -- graph data keys --
GD_CHG = "chg"
GD_AVG_NUM_UPCLIPS = "avg_num_upclips"
GD_AVG_NUM_DOWNCLIPS = "avg_num_downclips"
GD_AVG_UPCLIP_DED = "avg_upclip_ded"
GD_AVG_DOWNCLIP_DED = "avg_downclip_ded"

graph_data_keys =\
    [GD_CHG, GD_AVG_NUM_UPCLIPS, GD_AVG_NUM_DOWNCLIPS, GD_AVG_UPCLIP_DED,\
    GD_AVG_DOWNCLIP_DED]
# -- 
# - 

"""
Sets the random seed for all relevant libraries, including a potential
environment, to the specified value 'seed'.
"""
def set_random_seed(seed, env=None):
    if (env != None):
        env.seed(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# - keys of rollout entries -
# episode lengths in this rollout
RO_EP_LEN = 'ep_lens'
RO_EP_RET = 'ep_rets'
RO_OB = 'obs'
RO_AC = 'acs'
RO_ADV_GL = 'advs_gl'
RO_VAL_GL = 'vals_gl'
# - 

"""
Generic dataset with shuffle capabilities, for use in batch methods.
"""
class Dataset(object):
    def __init__(self, data_map):
        self.data_map = data_map
        self.num_data = next(iter(data_map.values())).shape[0]
        self.next_i = 0
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.num_data)
        np.random.shuffle(perm)
        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self.next_i = 0

    def iterate_once(self, batch_size):
        self.shuffle()

        # continue taking batches while there is enough remaining for a full
        # batch
        while (self.next_i <= (self.num_data - batch_size)):

            data_map = dict()
            for key in self.data_map:
                data_map[key] = self.data_map[key][self.next_i:self.next_i \
                    + batch_size]
            yield data_map

            self.next_i += batch_size

"""
Gets a tensor from a numpy array in a certain datatype.
"""
def from_numpy_dt(arr, in_device=device):
    return torch.from_numpy(arr).to(device=in_device, dtype=torch.float)

"""
Gets a detached numpy array from a tensor.
"""
def to_numpy_dt(tensor):
    return tensor.cpu().detach().numpy()

"""
Normal distribution with some additional features.
"""
class NormalUtil(normal.Normal):
    def get_prob_between(self, down, up):
        return self.cdf(up) - self.cdf(down)

    def get_prob_above(self, val):
        return 1 - self.cdf(val)

    def get_prob_below(self, val):
        return self.cdf(val)

def get_x_lim_mult(eps_mult, mean, mean_old, std):
    return ((mean ** 2 - mean_old ** 2) + (2 * (std ** 2) * torch.log(1 +\
            eps_mult))) / (2 * (mean - mean_old))

def get_x_lim(eps, up, mean, mean_old, std):
    mult = 1 if up else -1
    return get_x_lim_mult(mult * eps, mean, mean_old, std)

def get_discrepancy_and_penalty_contributions(eps_down, eps_up, dist, dist_old,
    std):
    x_up = get_x_lim(eps_up, True, dist.mean, dist_old.mean, std)
    x_down = get_x_lim(eps_down, False, dist.mean, dist_old.mean, std)

    int_1 = dist_old.get_prob_between(x_down, x_up)
    int_2 = dist.get_prob_between(x_down, x_up)
    int_3 = dist_old.get_prob_above(x_up)
    int_4 = dist.get_prob_above(x_up)

    discrepancy = eps_down + ((1 - eps_down) * int_1) - \
                (int_2 + ((eps_up + eps_down) * int_3))

    penalty_contributions = \
        (2 * int_4) - ((2 + eps_up - eps_down)* int_3) - eps_down - \
        ((1 - eps_down) * int_1) + int_2

    return discrepancy, penalty_contributions

def get_discrepancy_and_penalty_contribution_losses(eps_down, eps_up, dist,\
        dist_old, target_discrepancy, target_penalty_contribution, std):
    discrepancy, penalty_contributions =\
        get_discrepancy_and_penalty_contributions(eps_down, eps_up, dist,\
                dist_old, std)

    discrepancy_loss = (discrepancy - target_discrepancy) ** 2
    penalty_contribution_loss = (penalty_contributions -\
            target_penalty_contribution) ** 2
    return discrepancy_loss, penalty_contribution_loss, \
        discrepancy, penalty_contributions
