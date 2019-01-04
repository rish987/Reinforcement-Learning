# File: misc_utils.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Miscellaneous utility functions.
from imports import *
from constants import *

def clear_out_file():
    with open(OUTPUT_FILE, 'w+') as file:
        file.close()

"""
Prints a message the the terminal and saves it to an output file.
"""
def print_message(msg):
    with open(OUTPUT_FILE, 'a+') as file:
        file.write("{0}\n".format(msg))
    print(msg)


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
def from_numpy_dt(arr, device):
    return torch.from_numpy(arr).to(device, dtype=torch.float)

"""
Gets a detached numpy array from a tensor.
"""
def to_numpy_dt(tensor):
    return tensor.cpu().detach().numpy()
