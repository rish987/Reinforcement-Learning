# File: run.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/26/2018
# Description:
# Main runner class for implementation of OpenAI Proximal Policy Optimization 
# (PPO), using PyTorch instead of TensorFlow.
from imports import *
from misc_utils import set_random_seed
from ppo_model import PPOModel

# - hyperparameters -
# -- neural network parameters --
# number of nodes in each hidden layer
hidden_layer_size  =  64
# number of hidden layers
num_hidden_layers  =  2
# --

# -- run parameters --
# number of timesteps to train over
num_timesteps  =  1e6
# number of timesteps in a single rollout (simulated trajectory with fixed
# parameters)
timesteps_per_rollout = 2048
# random seed
seed = 0
# -- 

# epsilon as described by Schulman et. al.
clip_param = 0.2

# -- SGD parameters --
# number of training epochs per run
num_epochs = 10
# adam learning rate
alpha = 3e-4
# number of randomly selected timesteps to use in a single parameter update
batch_size = 64
# --

# -- GAE parameters --
# gamma and lambda factors used in Generalized Advantage Estimation (Schulman
# et. al.) to trade off between variance and bias in return/advantage
# approximation
gamma = 0.99
lambda_ = 0.95
# -- 
# - 

# TODO replace with passed-in environment
env_name = "InvertedPendulum-v2"

"""
Using the given ordered rollout data, gets the Generalized Advantage Estimation 
estimates for the advantage and value each timestep. This method trades off
between bias and variance.
"""
def get_adv_val_gl(rews, news, vals, nextval, gamma, lambda_,\
        timesteps_per_rollout):
    # - extend lists to length (timesteps_per_rollout + 1) -
    # okay to append 0 because if the last state was a terminal state, nextval
    # is 0
    news_aug = np.append(news, 0)
    vals_aug = np.append(vals, nextval)
    # - 

    # - populate advantages -
    advs_gl = np.zeros(timesteps_per_rollout)
    last_adv = 0.0
    for timestep in reversed(range(timesteps_per_rollout)):
        nonterminal_mult = 0 if news_aug[timestep + 1] else 1
        delta = rews[timestep] + gamma * vals_aug[timestep + 1] * \
            nonterminal_mult - vals_aug[timestep]

        advs_gl[timestep] = delta + gamma * lambda_ * nonterminal_mult * \
            last_adv
        last_adv = advs_gl[timestep]
    # -

    # get values
    vals_gl = advs_gl + vals

    return advs_gl, vals_gl

"""
Generator for length 'timesteps_per_rollout' rollouts under the given PPOModel
'model', operating within the environment 'env'.  
"""
def get_rollout(env, model, timesteps_per_rollout, gamma, lambda_):
    # - initialize current values -
    # currently on first timestep of episode?
    new = True
    # current observation
    ob = env.reset()
    # action taken at current state
    ac = None
    # current episode return
    ep_ret = 0
    # current episode length
    ep_len = 0
    # total total_timesteps rolled out over all runs
    total_timesteps = 0
    # - 

    # - initialize history arrays -
    # returns of all episodes in this rollout
    ep_rets = []
    # lengths of all episodes in this rollout
    ep_lens = []

    news = np.zeros(num_timesteps)
    obs = np.array([np.zeros((env.observation_space.shape,))] * num_timesteps)
    acs = np.array([np.zeros((env.action_space.shape,))] * num_timesteps)
    rews = np.zeros(num_timesteps)
    vals = np.zeros(num_timesteps)
    # - 

    # indefinitely continue generating rollouts when called
    while True: 
        # get the value of the current observation according to the model
        # TODO get both action and value at once to enhance parallelism?
        # TODO implement PPOModel.eval_value()
        val = model.eval_value(ob)

        # just completed a rollout
        if (total_timesteps > 0) and (total_timesteps % timesteps_per_rollout):
            nextval = 0.0 if new else val
            advs_gl, vals_gl = get_adv_val_gl(rews, news, vals, nextval, \
                    gamma, lambda_, timesteps_per_rollout)
            yield \
            {
                RO_EP_RET: ep_rets,
                RO_EP_LEN: ep_lens,
                RO_OB: obs,
                RO_AC: acs,
                RO_ADV_GL: advs_gl,
                RO_VAL_GL: vals_gl
            }
            ep_rets = []
            ep_lens = []

        # get the action that should be taken at the current observation
        # according to the model
        # TODO implement PPOModel.eval_policy()
        ac = model.eval_policy(ob)

        # timestep in this rollout
        timestep = (total_timesteps % timesteps_per_rollout)

        next_ob, rew, next_new, _ = env.step(ac)

        # - set history entries -
        news[timestep] = new
        obs[timestep] = ob
        acs[timestep] = ac
        rews[timestep] = rew
        vals[timestep] = val
        # -

        ob = next_ob
        new = next_new

        ep_ret += rew
        ep_len += 1

        # episode just finished
        if new:
            ep_rets.append(ep_ret)
            ep_lens.append(ep_len)
            ep_ret = 0
            ep_len = 0

            ob = env.reset()

        total_timesteps += 1
"""
Trains a PPO agent according to given parameters and reports results.
"""
def train():
    # - setup -
    # set up environment 
    env = gym.make(env_name)
    env.seed(seed)

    # set random seeds
    set_random_seed(seed, env)

    # create relevant PPO networks
    # TODO implement PPOModel ctor, pass in relevant parameters
    model = PPOModel()

    # total number of timesteps trained so far
    timesteps = 0

    # generator for getting rollouts
    # TODO implement get_rollout()
    rollout_gen = get_rollout(env, model, timesteps_per_rollout, gamma, lambda_)
    
    import sys
    sys.exit()
    # - 

    # - training -
    # continue training until timestep limit is reached
    while (timesteps < num_timesteps):
        # - SGD setup - 
        # get a rollout under this model for training
        rollout = rollout_gen.__next__()

        # update old policy function to new policy function
        # TODO implement PPOModel.update_old_pol()
        model.update_old_pol()

        # place data into dataset that will shuffle and batch them for training
        # TODO implement Dataset ctor
        data = Dataset(dict(ob=rollout[RO_OB], ac=rollout[RO_AC],\
            adv=rollout[RO_ADV_GL], val=rollout[RO_VAL_GL]))

        # linearly decrease learning rate
        alpha_decay_factor = \
                max(1.0 - float(timesteps) / num_timesteps, 0)
        # - 

        # - SGD training -
        # go through all epochs
        for _ in num_epochs:
            # go through all randomly organized batches
            # TODO implement Dataset.iterate()
            for batch in d.iterate(batch_size):
                # get gradient
                # TODO implement PPOModel.adam_update()
                model.adam_update(batch[RO_OB], batch[RO_AC],\
                    batch[RO_ADV_GL], batch[RO_VAL_GL])
        # - 

        # update total timesteps traveled so far
        timesteps += rollout[RO_EP_LEN].sum()
    # - 

def main():
    # TODO pass in necessary parameters (use no globals)
    train()

if __name__ == '__main__':
    main()
