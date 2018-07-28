# File: misc_utils.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Miscellaneous utility functions.
from imports import *

"""
Sets the random seed for all relevant libraries, including a potential
environment, to the specified value 'seed'.
"""
def set_random_seed(seed, env=None):
    if (env != None):
        env.seed(seed)

    # TODO need to set CUDA seed as well?
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
