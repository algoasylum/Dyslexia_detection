# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:46:48 2020

@author: Samik Pal
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn import preprocessing
import math
from scipy.fftpack import fft, fftfreq
from scipy.spatial.distance import euclidean as eu
import cv2
from random import randint
from sklearn.metrics.pairwise import cosine_similarity as cs

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
    

obsdata = [D_data[3], D_data[30],D_data[75],D_data[76],D_data[90],
           C_data[0], C_data[1],C_data[2],C_data[12],C_data[29]]
#1999 1999 1499 1499 1999
#1499 1999 1249 1749 999



D = [D_data[3], D_data[40],D_data[77],D_data[84],D_data[12]]
C = [C_data[1], C_data[34],C_data[16],C_data[86],C_data[52]]

y = C_data[30]
x = D_data[76]


 
def new_vals(data):
    x = [sum(x) for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x) for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    compl = np.array([complex(a,b) for a,b in zip(x, y)])
    
    mag = abs(compl)
    max_pos = np.where(mag == max(mag))
    max_com = compl[max_pos]
    normalized_complex = compl/max_com    
    
    return normalized_complex

def new_vals2(data):
    x = [sum(x) for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x) for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    mn = min([min(x), min(y)])
    mx = max([max(x), max(y)])
    
    xar = np.array(x)
    xar = xar - mn
    xar = xar / mx
       
    yar = np.array(y)
    yar = yar - mn
    yar = yar / mx
    
    compl = [complex(a,b) for a,b in zip(xar, yar)]
    
    return compl

def new_vals1(data):
    x = [sum(x) for x in zip(data['LX'].to_list(), data['RX'].tolist())]
    y = [sum(x) for x in zip(data['LY'].to_list(), data['RY'].tolist())]
    
    mn = min([min(x), min(y)])
    mx = max([max(x), max(y)])
    
    xar = np.array(x)
    xar = xar - mn
    xar = xar / mx
       
    yar = np.array(y)
    yar = yar - mn
    yar = yar / mx
    
    arr = np.row_stack((xar, yar))
    
    return arr

matrix_dd = np.empty([5,5])    
for a in range(5):
    for b in range(5):
        m = new_vals(D[a])
        n = new_vals(D[b])
        fft1 = fft(m)
        fft2 = fft(n)
        dis = eu([fft1], [fft2])
        matrix_dd[a][b] = dis 
        
matrix_cd = np.empty([5,5])    
for a in range(5):
    for b in range(5):
        m = new_vals(D[a])
        n = new_vals(C[b])
        fft1 = fft(m)
        fft2 = fft(n)
        dis = eu([fft1], [fft2])
        matrix_cd[a][b] = dis
        
matrix_cc = np.empty([5,5])    
for a in range(5):
    for b in range(5):
        m = new_vals(C[a])
        n = new_vals(C[b])
        fft1 = fft(m)
        fft2 = fft(n)
        dis = eu([fft1], [fft2])
        matrix_cc[a][b] = dis
        
mx = max([np.amax(matrix_dd), np.amax(matrix_cd), np.amax(matrix_cc)])
mn = min([np.amin(matrix_dd), np.amin(matrix_cd), np.amin(matrix_cc)])

img_dd = matrix_dd/mx  
img_dd = img_dd*256
      
img_cd = matrix_cd/mx  
img_cd = img_cd*256

img_cc = matrix_cc/mx  
img_cc = img_cc*256    


        
plt.imshow(img_dd, cmap='gray', vmin=0, vmax=256)
plt.show()


