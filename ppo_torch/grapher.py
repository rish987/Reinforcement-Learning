# File: grapher.py 
# Author(s): Rishikesh Vaishnav, TODO
# Created: 08/10/2018
import pickle
import numpy as np
import matplotlib.pyplot as plt;
plt.rc('text', usetex=True)

from misc_utils import \
    GD_CHG, GD_AVG_NUM_UPCLIPS, GD_AVG_NUM_DOWNCLIPS, GD_AVG_UPCLIP_DED,\
    GD_AVG_DOWNCLIP_DED, GD_EP_RETS, GD_TIMESTEPS, GD_ACT_UPCLIP_DED,\
    GD_ACT_DOWNCLIP_DED,\
    graph_data_keys

def graph_comp_ret_ded(data_contr, data_exp, graph_name, eps, ):
    iterations = np.arange(data_contr[GD_CHG].shape[0]) + 1

    plt.figure(figsize=(5, 10))
    plt.subplot(311)
    plt.ylabel("Average Return")
    plt.xlabel("Timestep")
    plt.ylim(env_to_range[graph_name])
    plt.title("Performance")

    plt.plot(data_contr[GD_TIMESTEPS], data_contr[GD_EP_RETS], linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        label='control')
    plt.plot(data_exp[GD_TIMESTEPS], data_exp[GD_EP_RETS], \
        linestyle='--', color=(0.0, 0.0, 0.0), \
        label='experimental')

    plt.legend()

    plt.subplot(312)
    plt.ylabel("Proportional Contribution")
    plt.xlabel("Number of Iterations")
    plt.title("Expected Penalty Contributions")

    plt.plot(iterations, data_contr[GD_AVG_UPCLIP_DED], linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        label='$1 - E[r_{t, CLIP}^+]$, control')
    plt.plot(iterations, data_contr[GD_AVG_DOWNCLIP_DED], linestyle='-', \
        color=(0.5, 0.5, 0.5), \
        label='$E[r_{t, CLIP}^-] - 1$, control')

    plt.plot(iterations, data_exp[GD_AVG_UPCLIP_DED], linestyle='--', \
        color=(0.0, 0.0, 0.0), \
        label='$1 - E[r_{t, CLIP}^+]$, experimental')
    plt.plot(iterations, data_exp[GD_AVG_DOWNCLIP_DED], linestyle='--', \
        color=(0.5, 0.5, 0.5), \
        label='$E[r_{t, CLIP}^-] - 1 $, experimental')

    plt.legend()
    
    plt.subplot(313)
    plt.ylabel("Loss Contribution")
    plt.xlabel("Number of Iterations")
    plt.title("Actual Penalty Contributions")

    plt.plot(iterations, data_contr[GD_ACT_UPCLIP_DED], linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        label='Positive Penalty Contribution, control')
    plt.plot(iterations, data_contr[GD_ACT_DOWNCLIP_DED], linestyle='-', \
        color=(0.5, 0.5, 0.5), \
        label='Negative Penalty Contribution, control')

    plt.plot(iterations, data_exp[GD_ACT_UPCLIP_DED], linestyle='--', \
        color=(0.0, 0.0, 0.0), \
        label='Positive Penalty Contribution, experimental')
    plt.plot(iterations, data_exp[GD_ACT_DOWNCLIP_DED], linestyle='--', \
        color=(0.5, 0.5, 0.5), \
        label='Negative Penalty Contribution, experimental')

    plt.legend()

    plt.tight_layout()

    plt.savefig("../notes/grapher/largebatch/eps_{0}/experiment_{1}.pgf".format(eps, graph_name))

def graph_ded_contr(data):
    iterations = np.arange(data[GD_CHG].shape[0]) + 1

    plt.figure()
    plt.ylabel("Proportional Contribution")
    plt.xlabel("Number of Iterations")
    plt.title("Expected Penalty Contributions")

    plt.plot(iterations, data[GD_AVG_UPCLIP_DED], linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        label='$1 - E[r_{t, CLIP}^+]$')
    plt.plot(iterations, data[GD_AVG_DOWNCLIP_DED], linestyle='-', \
        color=(0.5, 0.5, 0.5), \
        label='$E[r_{t, CLIP}^-] - 1$')

    plt.legend()

    plt.tight_layout()

    plt.savefig(GRAPH_OUT)
    plt.show()

def graph_chgs_and_clips(data):
    iterations = np.arange(data[GD_CHG].shape[0]) + 1

    plt.figure()
    ax = plt.subplot(211)
    plt.ylabel("count")
    plt.xlabel("Number of Iterations")
    plt.title("Clipping Behavior")

    plt.plot(iterations, data[GD_AVG_NUM_DOWNCLIPS], linestyle='-', \
        color=(0.0, 0.0, 0.0), \
        #label='$|\{r_t \mid r_t < 1 - \epsilon\}| - |\{r_t \mid r_t > 1 + \epsilon\}|$')
        label='$|\{t \mid (r_t > 1 + \epsilon) \land (G_t > 0)\}|$')
    ax.axhline(color="gray")

    plt.legend()
    
    plt.subplot(212)
    plt.ylabel("$||\\Delta \\pi||$")
    plt.xlabel("Number of Iterations")
    plt.title("Magnitude of Policy Change")
    plt.plot(iterations, data[GD_CHG], linestyle='-', color=(0.0, 0.0, 0.0),\
            label='$||\\Delta \\pi||$')

    plt.legend()

    plt.tight_layout()

    plt.savefig(GRAPH_OUT)
    plt.show()

environments_sub = ['InvertedPendulum-v2',\
    'InvertedDoublePendulum-v2', 'Hopper-v2',\
    'Swimmer-v2', 'Walker2d-v2']

# y-axis range for printing results from environment
env_to_range = {
        'InvertedPendulum-v2':(0, 1000),\
            'InvertedDoublePendulum-v2':(0, 200), 'Hopper-v2':(0, 1300),\
            'Swimmer-v2':(0, 60), 'Walker2d-v2':(0, 610)
        }

def main():
    for eps in [4, 3, 2, 1]:
        print(eps)
        for env_name in environments_sub:
            with open("data/largebatch/eps_{0}_data/data_contr_{1}.dat".format(eps, env_name), 'rb') as file:
                data_contr = pickle.load(file)
            with open("data/largebatch/eps_{0}_data/data_exp_{1}.dat".format(eps, env_name), 'rb') as file:
                data_exp = pickle.load(file)
            graph_comp_ret_ded(data_contr, data_exp, graph_name=env_name, eps=eps)

if __name__ == '__main__':
    main()
