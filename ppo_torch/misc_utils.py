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
