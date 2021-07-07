import numpy as np
import Data_load as dl

def conf_mat(act_lab, pred_lab):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Divides the entire data into groups of different lenghted vectors. Gives the confusion matrix based on the predictions for each separate group

#INPUT:
#     act_lab : set of actual labels obtained from the dataset
#     pred_lab : set of labels predicted by the classifier

# conf_lens: set of confusion matrix for different data lengths. 
#Structure:
#        [[[/,/],       -> for 999 length vectors
#          [/,/]],      

#         [[/,/],       -> for 999 length vectors
#          [/,/]],

#         [[/,/],       -> for 999 length vectors
#          [/,/]],

#         [[/,/],       -> for 999 length vectors
#          [/,/]],

#         [[/,/],       -> for 999 length vectors
#          [/,/]]]

#OUTPUT:  
#     conf_len: Confustion matrix of each section of the dataset, segregated based on the number of readings. This helps to access the performance of the classifier for each such section.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    data_lens = dl.data_lens()
    for a in range(len(act_lab)):
        conf_len[data_lens[a]][int(act_lab[a])][int(pred_lab[a])] += 1
    
    return np.array(conf_len)