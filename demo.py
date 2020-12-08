import glob
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing
from scipy.spatial.distance import euclidean as eu
import math
from scipy import signal

def create_bins(bins, fft, overlap_per):
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

def new_vals(data):
    x = [sum(x) for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x) for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    compl = np.array([complex(a,b) for a,b in zip(x, y)])
    
    return compl

def binning(D_in, bins, bins_per):
    all_buckets = []
    for dataset in D_in:
        for no in range(len(dataset)):
            d = new_vals(dataset[no])
            fft = np.fft.fft(d)   

            binned = binning(bins, fft, bins_per)
            all_buckets.append(binned)
            buckets = np.asarray(all_buckets)
    
    return buckets

def stft_run(D_in, n, f):
    C_spec = []
    for j in range(len(D_in)):
        data = D_in[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/125
        E =0#round((N*B - L)/(N-1))
        f, t, Zxx = signal.stft(D_in[j],fs= L/50, nperseg=B,noverlap= E,nfft=400)
        C_spec.append(np.abs(Zxx)**2)
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    ret = []
    for i in range(len(vec)):
        ret.append(vec[i].flatten())
    print(vec)
    return ret

def run_kmeans(data, labels, n, r_state):
    kmeans = KMeans(n_clusters = n, random_state= r_state).fit(data)
    predicted_labels = kmeans.labels_
    actual_labels = labels
    conf_m = confusion_matrix(actual_labels,predicted_labels)[:2]
    acc = accuracy_score(actual_labels,predicted_labels)*100
    
    return conf_m, acc