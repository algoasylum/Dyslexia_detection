import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as eu
from scipy.spatial.distance import cosine 
import math
from scipy import signal

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import preprocessing

#Get Control and Dyslexic data for 'X' and 'Y'
def get_data():
    D_path = glob.glob('Data\Dyslexic' + "\*")
    C_path = glob.glob('Data\Control' + "\*")

    C_data = []
    for path in C_path:
        temp = pd.read_csv(path)
        temp = temp.drop('Unnamed: 0',axis = 1)
        C_data.append(temp)

    D_data = []
    for path in D_path:
        temp = pd.read_csv(path)
        temp = temp.drop('Unnamed: 0',axis = 1)
        D_data.append(temp)

    return C_data, D_data

#Get Control and Dyslexic data as required for the STFT
def get_stft_data(C_data, D_data): 
    C_new = []
    for data in C_data:
        X =data[['LX','RX']]
        Y =data[['LY','RY']]
        Xm = X.mean(axis=1)
        Ym = Y.mean(axis=1)
        f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
        f = f.transpose()
        f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
        C_new.append(f)

    D_new = []
    for data in D_data:
        X =data[['LX','RX']]
        Y =data[['LY','RY']]
        Xm = X.mean(axis=1)
        Ym = Y.mean(axis=1)
        f = pd.DataFrame([data.iloc[:,0],Xm,Ym])
        f = f.transpose()
        f = f.rename(columns = {'Unnamed 0': 'X', 'Unnamed 1': 'Y'})
        D_new.append(f)
    
    C_new,D_new = normalise_data(C_new,D_new)
    
    C_cmx = []
    C_real= []
    C_img=[]
    for j in range(len(C_new)):
        dat = C_new[j]
        x = dat['X']
        y = dat['Y']
        t = dat['T']

        z=[]
        x_in=[]
        y_in=[]
        for i in range(0,x.size):
            z.append(complex(x[i],y[i]))
            x_in.append(x[i])
            y_in.append(y[i])


        C_cmx.append(z)
        C_real.append(x_in)
        C_img.append(y_in)

    D_cmx = []
    D_real= []
    D_img=[]
    for j in range(len(D_new)):
        dat = D_new[j]
        x = dat['X']
        y = dat['Y']
        t = dat['T']

        z=[]
        x_in=[]
        y_in=[]
        for i in range(0,x.size):
            z.append(complex(x[i],y[i]))
            x_in.append(x[i])
            y_in.append(y[i])
        D_cmx.append(z)
        D_real.append(x_in)
        D_img.append(y_in)
    
    return C_cmx, C_real, C_img, D_cmx, D_real, D_img, C_new, D_new
    
    
def normalise_data(C_new,D_new): 
    for i in range(len(C_new)):
        C_tempx = np.abs(C_new[i]['X'])
        mx = max(C_tempx)
        C_tempy = np.abs(C_new[i]['Y'])
        my= max(C_tempy)
        C_new[i]['X'] = C_new[i]['X']/np.abs(mx)
        C_new[i]['Y'] = C_new[i]['Y']/np.abs(my)
    for i in range(len(D_new)):
        D_tempx = np.abs(D_new[i]['X'])
        mx = max(D_tempx)
        D_tempy = np.abs(D_new[i]['Y'])
        my= max(D_tempy)
        D_new[i]['X'] = D_new[i]['X']/np.abs(mx)
        D_new[i]['Y'] = D_new[i]['Y']/np.abs(my)  
    return C_new,D_new
    
    
def average_l_r(data):
    x = [sum(x)/2 for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x)/2 for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    return x, y

def data_lens():
    C_data, D_data = get_data()
    lens = []
    for dSet in [C_data, D_data]:
        for data in dSet:
            lens.append(int(((len(data['LX']) + 1)/250) - 4))
    return lens