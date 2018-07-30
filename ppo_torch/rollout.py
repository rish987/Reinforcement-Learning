# File: rollout.py 
# Author(s): Rishikesh Vaishnav
# Created: 07/27/2018
# Description:
# Functions relating to environment rollouts.
from imports import *
from misc_utils import RO_EP_LEN, RO_EP_RET, RO_OB, RO_AC, RO_ADV_GL, RO_VAL_GL

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

    news = np.zeros(timesteps_per_rollout)
    obs = np.array([np.zeros(env.observation_space.shape[0])] * \
        timesteps_per_rollout)
    acs = np.array([np.zeros(env.action_space.shape[0])] * \
        timesteps_per_rollout)
    rews = np.zeros(timesteps_per_rollout)
    vals = np.zeros(timesteps_per_rollout)
    # - 

    # indefinitely continue generating rollouts when called
    while True: 
        # get the value of the current observation according to the model
        # TODO get both action and value at once to enhance parallelism?
        val = model.eval_value(from_numpy_dt(ob)).detach().numpy()

        # just completed a rollout
        if (total_timesteps > 0) and ((total_timesteps %\
            timesteps_per_rollout) == 0):
            nextval = 0.0 if new else val
            advs_gl, vals_gl = get_adv_val_gl(rews, news, vals, nextval, \
                    gamma, lambda_, timesteps_per_rollout)
            # TODO convert all to floats?
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
        ac = model.eval_policy_var(from_numpy_dt(ob)).detach().numpy()

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
