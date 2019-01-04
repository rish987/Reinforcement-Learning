# File: constants.py 
# Author(s): Rishikesh Vaishnav
# Created: 26/07/2018
# Description:
# Constant variables.

# - graphing parameters -
# default graph output file
GRAPH_OUT = "graph_out.pgf"

# -- graph data keys --
GD_CHG = "chg"
GD_AVG_NUM_UPCLIPS = "avg_num_upclips"
GD_AVG_NUM_DOWNCLIPS = "avg_num_downclips"
GD_AVG_UPCLIP_DED = "avg_upclip_ded"
GD_AVG_DOWNCLIP_DED = "avg_downclip_ded"
GD_EP_RETS = "ep_rets"
GD_TIMESTEPS = "timesteps"
GD_ACT_UPCLIP_DED = "act_upclip_ded"
GD_ACT_DOWNCLIP_DED = "act_downclip_ded"

graph_data_keys =\
    [GD_CHG, GD_AVG_NUM_UPCLIPS, GD_AVG_NUM_DOWNCLIPS, GD_AVG_UPCLIP_DED,\
    GD_AVG_DOWNCLIP_DED, GD_EP_RETS, GD_TIMESTEPS, GD_ACT_UPCLIP_DED,\
    GD_ACT_DOWNCLIP_DED]
# -- 
# - 

# file for text output
OUTPUT_FILE = "temp_out"

# - keys of rollout entries -
# episode lengths in this rollout
RO_EP_LEN = 'ep_lens'
RO_EP_RET = 'ep_rets'
RO_OB = 'obs'
RO_AC = 'acs'
RO_ADV_GL = 'advs_gl'
RO_VAL_GL = 'vals_gl'
# - 

