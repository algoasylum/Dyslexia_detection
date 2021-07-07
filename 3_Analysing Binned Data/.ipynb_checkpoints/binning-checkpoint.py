import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing

import Data_load as dl

def binning(bins, fft, overlap_per):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# maps a vector any lenght to a vector of a fixed length(bins) as needed. 

#INPUT: 
#     bins: Number of elements in the target vector
#     fft: Set of all FFTs
#     overlap_per: Percentage of overlap between successive entries

#Each entry of the resulting vector is a sum of fixed number of elements of the input vector. Few of these elements are considered common for successive entries into the resulting vector. This is the overlapping factor.

# So the fixed number of elements considered for the each entry = (lenght of input vector/lenght of output vector) + overlap

#OUTPUT: 
#     binned: list of all ffts after binning
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    div_size = len(fft)/bins
    bin_size = div_size*(1+(overlap_per/100))
    half_bin = bin_size/2
    
    binned = []
    
    current_step = bin_size
    for a in range(bins):
        
        pos = np.ceil(half_bin + a*(div_size))
        start = 0 if a == 0 else int(np.ceil(pos - half_bin))
        end = -1 if a == (bins-1) else int(np.ceil(pos + half_bin))
        #print([start, end])
        
        binned = np.append(binned, sum(np.abs(fft[start : end]))) 
        
    return binned


def kmeans_binning(D_data,C_data, bins, overlap_per):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# does the binning process and then applies kmeans to the binnned vectors. 
#INPUT: 
#     D_data: The data of all Dyslexic candidates
#     C_data: The data of all Control candidates
#     fft: Set of all FFTs
#     overlap_per: Percentage of overlap between successive entries

# OUTPUT:
#     1. conf_len: divides the entire data into groups of different lenghted vectors. Gives the confusion matrix based on the predictions for each separate group
#     2. conf_m: gives confusion matrix based on prediction for the entire dataset 
#     3. acc: gives accuracy of predictions
#     4. buckets: gives the binned data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data_lens = dl.data_lens()
    data_sets = [D_data,C_data]
    all_buckets = []
    conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    
    for dataset in data_sets:
        for no in range(len(dataset)):
            d = dl.average_l_r(dataset[no])
            fft = np.fft.fft(d)   

            binned = binning(bins, fft, overlap_per)
            all_buckets.append(binned)
    buckets = np.asarray(all_buckets)
            
    kmeans = KMeans(n_clusters = 2, random_state=0).fit(buckets)

    predicted_labels = kmeans.labels_
    actual_labels = np.concatenate((np.ones(98), np.zeros(88)))
    
    for a in range(len(buckets)):
        conf_len[data_lens[a]][int(actual_labels[a])][int(predicted_labels[a])] += 1
    
    conf_m = confusion_matrix(actual_labels,predicted_labels)[:2]
    acc = accuracy_score(actual_labels,predicted_labels)*100
    
    return conf_len, conf_m, acc, buckets


def sectional_kmeans(n, buckets):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#divides each entry into different sections and then takes a kmeans for each section separately. 

#INPUT: 
#     n: Number of sections
#     buckets: Dataset 

# OUTPUT:
#     1. all_sec_len - gives the length wise confusion matrix for each section.
#     2. all_sec_mat - gives the confusion matrix for each section.
#     3. all_sec_acc - accuracy values for each section.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dyx = len(buckets) - 88
    l = len(buckets[0])
    all_sec_mat = []
    all_sec_acc = []
    all_sec_len = []
    data_lens = dl.data_lens()
    
    for section in range(int(l/n)):
        secs = []
        for cand in buckets:
            secs.append(cand[section*n:(section+1)*n])
        kmeans = KMeans(n_clusters = 2, random_state=0).fit(secs)
        predicted_labels = kmeans.labels_
        actual_labels = np.concatenate((np.ones(dyx), np.zeros(88)))
        all_sec_acc.append(accuracy_score(actual_labels,predicted_labels)*100)
        
        conf_len = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
        for a in range(len(actual_labels)):
            conf_len[data_lens[a]][int(actual_labels[a])][int(predicted_labels[a])] += 1

        all_sec_len.append(np.array(conf_len))
        all_sec_mat.append(confusion_matrix(actual_labels,predicted_labels)[:2])
        
        
    return all_sec_len, all_sec_mat, all_sec_acc

def new_vals(data):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Combines the X and Y axis average values of the reading of a single candidate into a complex vector where the real part represents the average of x axis readings and the imaginary part represents the average of y axis readings. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    x = [sum(x)/2 for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x)/2 for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    compl = np.array([complex(a,b) for a,b in zip(x, y)])
    
    return compl