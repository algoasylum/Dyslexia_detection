import numpy as np
from scipy.spatial.distance import euclidean as eu
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure

def corr(buckets, principalComponents):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#gives correlation of each dimention of the data with the first principal component. 
#INPUT:
#     buckets: dataset containing entry of each candidate whose corelations need to be found out
#     principalComponents: the principal components obtained by taking a Principal Component Analysis.

#It'll compile the first dimension of each entry into a vector and then take it's corelation with the first principal component. It'll do the same with the second thrid and so on and store the coorelation values of each dimension into a single vector. 

#OUTPUT:
#     cor: Set of all corelation values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cor = []
    for dim in buckets.T:
        cor.append(np.corrcoef(dim, principalComponents[0])[1,0])
    
    return cor

def apply_wts(wts, binned):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#multiplies weights for each dimension to each dimension of each entry
#INPUT:
#     wts: set of weights for each dimension 
#     binned: dataset containing entry of each candidate to which weights need to be applied to

#OUTPUT:
#     wtd_bins: dataset after applying weights
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    wtd_bins = []
    for dim in range(len(binned.T)):
        wtd_bins.append(wts[dim]*binned.T[dim])
    wtd_bins = np.array(wtd_bins).T
    
    return wtd_bins

def distance_from_cluster_center(centers, data):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#gives the distance of each vector from the cluster center

#INPUT:
#     centers: list of the 2 cluster centers
#     data: set of vectors who's distance from the cluster centers is to be found

#OUTPUT:
#     distances: List of the distances of each vector from each cluster center
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    distances = []
    for a in range(186):
        distances.append([eu(centers[0], data[a]), eu(centers[1], data[a]), min(eu(centers[0], data[a]), eu(centers[1], data[a]))])

    d_max = np.amax(distances)
    distances = np.array(distances)/d_max
    
    return distances


def degree_of_classification(distances, act_label, pred_label):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gives the difference between the distances from each cluster center. It does this for the correctly classified and wrongly classified separately 
#INPUT: 
#     distances: set of distances of each vector from each cluster center
#     act_label: the correct labels obtained from the dataset
#     pred_label: the labels predicted by the classifier/machine learning model 

#OUTPUT: 
#     dif_corr: vector of difference between the distances from each cluster center for the correctly predicted entries
#     wr_corr: vector of difference between the distances from each cluster center for the wrongly predicted entries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dif_corr = []
    dif_wr = []
    for a in range(186):
        if act_label[a] == [1-a for a in pred_label][a]:
            dif_corr.append(np.abs(distances[a][0] - distances[a][1]))
        else:
            dif_wr.append(np.abs(distances[a][0] - distances[a][1]))
            
    return dif_corr, dif_wr
            
def plot_degree(dif_corr, dif_wr):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Plots the digree of classification in the following manner:
#1. Correctly classified in Ascending order
#2. Wrongly classified in Ascending order
#3. Correctly classified in serial order
#4. Wrongly classified in serial order
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
