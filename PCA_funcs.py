import numpy as np
from scipy.spatial.distance import euclidean as eu
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure

def corr(buckets, principalComponents):
    cor = []
    for dim in buckets.T:
        cor.append(np.corrcoef(dim, principalComponents[0])[1,0])
    
    return cor

def apply_wts(wts, binned):
    wtd_bins = []
    for dim in range(len(binned.T)):
        wtd_bins.append(wts[dim]*binned.T[dim])
    wtd_bins = np.array(wtd_bins).T
    
    return wtd_bins

def distance_from_cluster_center(centers, data):
    distances = []
    for a in range(186):
        distances.append([eu(centers[0], data[a]), eu(centers[1], data[a]), min(eu(centers[0], data[a]), eu(centers[1], data[a]))])

    d_max = np.amax(distances)
    distances = np.array(distances)/d_max
    
    return distances


def degree_of_classification(distances, act_label, pred_label):
    dif_corr = []
    dif_wr = []
    for a in range(186):
        if act_label[a] == [1-a for a in pred_label][a]:
            dif_corr.append(np.abs(distances[a][0] - distances[a][1]))
        else:
            dif_wr.append(np.abs(distances[a][0] - distances[a][1]))
            
    return dif_corr, dif_wr
            
def plot_degree(dif_corr, dif_wr):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    fig.subplots_adjust(hspace=0.3, wspace = 0.1)


    metrics = [dif_corr, dif_wr]
    plot = 0
    ax = axes.flatten()
    for row in range(2):
        for col in range(2):
            ax[plot].plot(metrics[col] if row else sorted(metrics[col]), 'o', markersize=2)
            plot += 1   

    ax[0].set_title('1. Correctly classified in Ascending order')
    ax[1].set_title('2. Wrongly classified in Ascending order')
    ax[2].set_title('3. Correctly classified in serial order')
    ax[3].set_title('4. Wrongly classified in serial order')
