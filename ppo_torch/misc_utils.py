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
    torch.cuda.manual_seed_all(seed)
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

"""
Normalization utility class; normalizes values on a running basis.
"""
class Normalizer(object):
    def __init__(self, shape):
        self.epsilon = 1e-4
        self.val_clip = 10

        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")

        self.count = self.epsilon

    """
    Update mean, variance, and count, using given incremental value.
    """
    def update(self, val):
        # calculate intermediate values
        new_count = self.count + 1
        delta = val - self.mean
        M2 = (self.var * self.count) + (np.square(delta) * \
                (self.count / new_count))

        # reset running values
        self.mean = self.mean + delta / new_count
        self.var = ((self.var * self.count) + (np.square(delta) * \
                (self.count / new_count))) / new_count
        self.count = new_count

    def shift(self, val):
        return val - self.mean

    def scale(self, val):
        return val / np.sqrt(self.var + (self.epsilon ** 2))

    def clip(self, val):
        return np.clip(val, -self.val_clip, self.val_clip)

    """
    Get the normalized value according to the current mean and standard
    deviation.
    """
    def normalize(self, val):
        return self.clip(self.scale(self.shift(val)))

    """
    Update the running values according to the single value, and
    return the normalized value.
    """
    def update_and_normalize(self, val):
        self.update(val)
        return self.normalize(val)

"""
Generic wrapper for environments, intended for extension.
"""
class EnvWrapper(object):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def seed(self, seed):
        return self.env.seed(seed)

    def step(self, ac):
        return self.env.step(ac)

    def reset(self):
        return self.env.reset()

"""
Normalization wrapper for environments; normalizes observations and returns
based on those seen in training period.
"""
class EnvNormalized(EnvWrapper):
    def __init__(self, env):
        super(EnvNormalized, self).__init__(env)

        self.gamma = 0.99

        # discounted return of current timestep
        self.ret = np.zeros(())

        self.obs_norm = Normalizer(self.observation_space.shape)
        self.ret_norm = Normalizer(())

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)

        # get this timestep's discounted return
        self.ret = rew + (self.gamma * self.ret)
        self.ret_norm.update(self.ret)

        return self.obs_norm.update_and_normalize(obs), \
            self.ret_norm.scale(rew), done, info

    def reset(self):
        self.ret = np.zeros(())
        return self.obs_norm.update_and_normalize(self.env.reset())

"""
Initializes the given neural network layer using an orthogonal distribution for
the weights and zeroes for the biases.
"""
def layer_init(module):
    nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
    nn.init.constant_(module.bias.data, 0.0)
    return module
