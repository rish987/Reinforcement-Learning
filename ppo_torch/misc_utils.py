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
