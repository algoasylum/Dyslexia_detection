import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
from scipy.spatial.distance import euclidean as eu
from scipy.spatial.distance import cosine 
import math
from scipy import signal

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: 'tmat' is referenced as input to most functions
# tmat: Array of all complex input time-series of length 186. 88 Control Group Readings. 98 Dyslexic group readings.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

y= np.concatenate((np.ones(88), np.zeros(98)))

#Standard STFT Run. Returns Full Flattened Vector
def stft_run(tmat, n_ratio,o_ratio):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Standard STFT Run:
# Input:
#     tmat: Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. Bin width (B) = Length (L)/n_ratio
#     o_ratio: Ratio of Bin_width to Overlap


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) Shape: (t*f)x1 : Flattened C_spec vector. 


# NOTE: All the modifications in further functions is on "C_spec --> vec" process 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        
        
        C_spec.append(np.abs(Zxx)**2)
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    print(C_spec[i].shape)
    
    for i in range(len(C_spec)):
        vec[i]=vec[i].flatten()
    
    

    return vec

#Considers Bins till 'lim'. Returns flattened vector for only some bins
def stft_run_half(tmat, n_ratio,o_ratio,lim1,lim2):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculates STFT of selected temporal bins. 
# ------------------------------------------
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# |           ||||||||||||||||             |
# ------------------------------------------
#           lim1            lim2

# Input:
#     tmat   : Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. Bin width (B) = Length (L)/n_ratio
#     o_ratio: Ratio of Bin_width to Overlap
#     lim1   : Lower limit of selected bins
#     lim2   : Upper limit of selected bins


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) : Flattened C_spec vector sliced by bins 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lim1= int(lim1)
    lim2= int(lim2)
    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        half_im=[]
        for i in range(len(tot)):
            half_im.append(tot[i][lim1:lim2])
            
        
        C_spec.append(np.asarray(half_im))
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    #print(C_spec[i].shape)
    
    for i in range(len(C_spec)):
        vec[i]=vec[i].flatten()
    
    

    return vec

#Flattened vector depending on frequency
def stft_run_freq(tmat, n_ratio,o_ratio,lim1,lim2):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculates STFT of selected temporal bins. 
# ------------------------------------------
# |========================================| lim2
# |========================================|
# |========================================| lim1
# |                                        |
# |                                        |
# |                                        |
# ------------------------------------------


# Input:
#     tmat   : Input data control+dyslexic
#     n_ratio: Ratio to equalise Output length. Bin width (B) = Length (L)/n_ratio
#     o_ratio: Ratio of Bin_width to Overlap
#     lim1   : Lower limit of selected frequency range
#     lim2   : Upper limit of selected frequency range


# Zxx (Complex) Shape: t x f: Compute 2D STFT output array. 
#                             --> f: proportional to (Length of signal / Bin Width)
#                             --> t: depends on Bin Width an Overlap .... (Refer binning approach for exact calcualtion)
# C_spec (Real) Shape: t x f: Zxx converted to absolute values.


# OUTPUT:
#      vec (Real) : Flattened C_spec vector sliced by frequency 



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    lim1= int(lim1)
    lim2= int(lim2)
    C_spec = []
    vec= []
    for j in range(len(tmat)):
        data = tmat[j]
        L= len(data)+1
        k = int((len(data) + 1)/250) # k varies from 4 to 8
        N=20
        B  = L/n_ratio
        E =B/o_ratio#round((N*B - L)/(N-1))
        nf = 2000/n_ratio
        f, t, Zxx = signal.stft(tmat[j],fs= L/250, nperseg=B,noverlap= E,nfft=nf)
        
        tot = np.abs(Zxx)**2
        
        
        C_spec.append(np.abs(Zxx)**2)
    
    vec = []
    for i in range(len(C_spec)):
        vec.append(C_spec[i])
    print(C_spec[i].shape)
    
    factor = len(C_spec[i][3])
    l1= factor*lim1
    l2 = factor*lim2
    print(l1,l2)
    lfvec =[]
    hfvec =[]
    fvec=[]
    for i in range(len(C_spec)):
        x=vec[i].flatten()
        #print(len(x))
        #lfvec.append(x[:lim])
        #hfvec.append(x[lim:])
        fvec.append(x[l1:l2])
    
    

    return fvec

#Final Testing result
def final(X_train,y_train,X,y):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to plot final perceptron results.

# scatter plot of all reading with heights from x-axis as distances from the separating plane.
# INPUT:
#      X_train, y_train: Training set readings and labels
# OUTPUT:
#      ht : Distances form separating plane ( +ve: Control side | -ve: Dyslexic side)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    clf.fit(X_train, y_train)
    ht = clf.decision_function(X)
    x = range(len(y))
    fig, ax = plt.subplots()
    ax.scatter(x,ht,c = ylen)
    ax.axvline(x=88, color='b', linestyle='-')
    ax.axhline(y=0, color='r', linestyle='-')
    print(clf.score(X,y))
    return ht

#Training testing data based on fixed index defined signals
def create_train_test_set(X,y,index):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to split training and testing set for all perceptron runs.
#INPUT:
#      X: Set of all Control and Dyslexic readings
#  index: Indices of the fixed training set 
# OUTPUT:
#      X_train, X_test,y_train,y_test,index_train,index_test : 
#      Training and Testing sets of : 
#                                   X: readings
#                                   y: Labels
#                               index: Indices

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    index_test =[]
    index_train=index
    for i in range(len(X)):
        if i in index_train:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
            index_test.append(i)
    return X_train, X_test,y_train,y_test,index_train,index_test


def get_misclassified(res):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get misclassified points form final result 
#INPUT:
#     Res: Distances from separating plane
#  
# OUTPUT:
#      WrongClass: Array of all wrongly classified points
#                  Every element is [index, Length of reading]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    WrongClass=[]
    for i in range(88):
        if(res[i]<0):
            WrongClass.append([i,ylen[i]])
    for i in range(88,186):
        if(res[i]>0):
            WrongClass.append([i,ylen[i]])
    return WrongClass
    