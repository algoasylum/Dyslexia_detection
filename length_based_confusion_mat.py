import numpy as np
import Data_load as dl

def conf_mat(act_lab, pred_lab):
    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    data_lens = dl.data_lens()
    for a in range(len(act_lab)):
        conf_len[data_lens[a]][int(act_lab[a])][int(pred_lab[a])] += 1
    
    return np.array(conf_len)