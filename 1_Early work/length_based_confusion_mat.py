import numpy as np
import Data_load as dl

def conf_mat(act_lab, pred_lab):
#divides the entire data into groups of different lenghted vectors. Gives the confusion matrix based on the predictions for each separate group
    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    data_lens = dl.data_lens()
    for a in range(len(act_lab)):
        conf_len[data_lens[a]][int(act_lab[a])][int(pred_lab[a])] += 1
    
    return np.array(conf_len)