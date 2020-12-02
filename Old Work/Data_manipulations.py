# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:07:37 2020

@author: Samik Pal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn import preprocessing
import math

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
    

feature_list=['LX', 'LY', 'RX', 'RY']

obsdata = [D_data[3], D_data[30],D_data[75],D_data[76],D_data[90],C_data[0], C_data[1],C_data[2],C_data[12],C_data[29]]
    
list_DF = D_data + C_data

fig_names = ['D_Data','D_Data','D_Data','D_Data','D_Data','C_Data','C_Data','C_Data','C_Data','C_Data']

def get_padded(a,ml):
    a = np.pad(a, (0,abs(ml - len(a))), 'mean')
    
    return a 

def get_padded2(a, ml):
    
    while a.shape[0] < 1999:
        
        padded_a = []
        
        diff = ml - len(a)
    
        a_m = np.array(a)
        per_no = math.ceil(len(a) / diff)
        cnt = 0
    
        prev = a_m[0].tolist()
    
        for i in a_m:
            if (cnt != (per_no - 1)):
                padded_a.append(i.tolist())
                cnt += 1
            else:        
                ap = []
                for j in range(5):
                    ap = np.append(ap, (prev[j] + i[j])/2)
    
                padded_a.append(ap.tolist())
                padded_a.append(i.tolist())
                cnt = 0
    
            prev = i 
    
        padded_a.append(prev.tolist())
        
        a = np.array(padded_a)
    
    
    return a

def padding_at_last(list_DF , feature_list):
    subject = []
    for l in list_DF:
        
        _temp = []
        for f in feature_list:
            _temp+=l[f].tolist()
            
        subject.append(_temp)
    
    padded_subject = []
    for s in subject:
        padded_subject.append(get_padded(s,7996))        
    padded_subject = np.asarray(padded_subject)
    return padded_subject

    
def padding_in_bw(data_sets, featuresets):
    subjects = [] 
    for ds in data_sets:
        
        _temp = []
        for f in featuresets:
            _temp += get_padded(ds[f].tolist(), 1999).tolist()
            
        subjects.append(_temp)
        
    subjects = np.asarray(subjects)
    return subjects

def interpolating(data_set):
    subject = []
    for l in data_set:
        arr = np.array(l)

        if arr.shape[0] < 1999:
            arr = get_padded2(arr, 1999)
        arr = np.delete(arr, 0, axis = 1)

        arr = arr.transpose().reshape(1,7996)
        subject.append(arr[0].tolist())

    subject = np.array(subject)
    
    return subject

def original_vectors(data_sets, featuresets):
    subjects = [] 
    for ds in data_sets:
        
        _temp = []
        for f in featuresets:
            _temp += ds[f].tolist()
            
        subjects.append(_temp)
        
    subjects = np.asarray(subjects)
    return subjects


def plot_all(og, at_end, in_bw, inter, names):
    c = 1
    for o,a,ib,i,text in zip(og, at_end, in_bw, inter, names):
        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle(text + "->" + str(len(o)))
        
        
        
        vals = np.array(o)
        mn = min(vals)
        vals = vals - mn
        mx = max(vals)
        vals = vals/mx
        ax1.axvline(x = len(o)/4, color = 'r', linewidth= 0.1)
        ax1.axvline(x = 2*(len(o)/4), color = 'r', linewidth= 0.1)
        ax1.axvline(x = 3*(len(o)/4), color = 'r', linewidth= 0.1)
        ax1.plot(range(len(o)), vals, linewidth= 0.3) 
        
        
        vals = np.array(a)
        mn = min(vals)
        vals = vals - mn
        mx = max(vals)
        vals = vals/mx
        ax2.axvline(x = len(o)/4, color = 'r', linewidth= 0.1)
        ax2.axvline(x = 2*(len(o)/4), color = 'r', linewidth= 0.1)
        ax2.axvline(x = 3*(len(o)/4), color = 'r', linewidth= 0.1)
        ax2.axvline(x = len(o), color = 'r', linewidth= 0.1)
        ax2.plot(range(7996), vals, linewidth= 0.3) 
        
        for f, ax in zip([ib, i], [ax3,ax4]):
            vals = np.array(f)
            mn = min(vals)
            vals = vals - mn
            mx = max(vals)
            vals = vals/mx
            ax.axvline(x = 7996/4, color = 'r', linewidth= 0.1)
            ax.axvline(x = 2*(7996/4), color = 'r', linewidth= 0.1)
            ax.axvline(x = 3*(7996/4), color = 'r', linewidth= 0.1)
            ax.plot(range(7996), vals, linewidth= 0.3)
            
                        
        plt.savefig(text + '_' + str(c)+ ' ' + str(len(o)) +".png", dpi = 1200)

        c = 1 if c == 5 else (c + 1)
        
        
def plot_all_diff(og, at_end, in_bw, inter, names):
    c = 1
    for o,a,ib,i,text in zip(og, at_end, in_bw, inter, names):
        
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #fig.suptitle(text + "->" + str(len(o)))
        
        vals = np.array(o)
        mn = min(vals)
        vals = vals - mn
        mx = max(vals)
        vals = vals/mx
        
        plt.figure()
        plt.axvline(x = len(o)/4, color = 'r', linewidth= 0.1)
        plt.axvline(x = 2*(len(o)/4), color = 'r', linewidth= 0.1)
        plt.axvline(x = 3*(len(o)/4), color = 'r', linewidth= 0.1)
        fig = plt.figure()
        plt.plot(range(len(o)), vals, linewidth= 0.3) 
        fig.suptitle(text + '_' + str(c)+ '_o_' + str(len(o)))
        plt.savefig(text + '_' + str(c)+ '_o_' + str(len(o)) +".png", dpi = 1200)
        
        
        
        vals = np.array(a)
        mn = min(vals)
        vals = vals - mn
        mx = max(vals)
        vals = vals/mx
        
        plt.figure()
        plt.axvline(x = len(o)/4, color = 'r', linewidth= 0.1)
        plt.axvline(x = 2*(len(o)/4), color = 'r', linewidth= 0.1)
        plt.axvline(x = 3*(len(o)/4), color = 'r', linewidth= 0.1)
        plt.axvline(x = len(o), color = 'r', linewidth= 0.1)
        fig = plt.figure()
        plt.plot(range(7996), vals, linewidth= 0.3) 
        fig.suptitle(text + '_' + str(c)+ '_a_' + str(len(o)))
        plt.savefig(text + '_' + str(c)+ '_a_' + str(len(o)) +".png", dpi = 1200)
        
        for f, n in zip([ib, i], ['_ib_','_i_']):
            vals = np.array(f)
            mn = min(vals)
            vals = vals - mn
            mx = max(vals)
            vals = vals/mx
            
            plt.figure()
            plt.axvline(x = 7996/4, color = 'r', linewidth= 0.1)
            plt.axvline(x = 2*(7996/4), color = 'r', linewidth= 0.1)
            plt.axvline(x = 3*(7996/4), color = 'r', linewidth= 0.1)
            fig = plt.figure()
            plt.plot(range(7996), vals, linewidth= 0.3)
            fig.suptitle(text + '_' + str(c)+ n + str(len(o)))
            plt.savefig(text + '_' + str(c)+ n + str(len(o)) +".png", dpi = 1200)
            
                        
        #plt.savefig(text + '_' + str(c)+ ' ' + str(len(o)) +".png", dpi = 1200)

        c = 1 if c == 5 else (c + 1)


def remove_outliers(data_set):
    for vector in data_set:
        vector = list(vector)
        ln = len(vector)
        n = int(ln/4)
        features = [vector[:ln][i:i + n] for i in range(0, ln, n)]
        print(len(features))
        cnt = 0 
        for f in features:
            
            dif = []
            for i in range(1, len(f)-1):
                dif.append(abs(((f[i-1] + f[i+1])/2) - f[i]))
                        
            #print(type(dif))
            #print(type(list(f)))
            
            dif.insert(0, dif[0])
            '''
            dif.insert(0, 1)
            dif.insert(0, 2)
            dif.insert(0,len(f) - 3)
            dif.insert(0,len(f) - 2)
            '''
            dif.insert(len(f), dif[-1])
            dif = list(dif)
            
            d = dict(zip(list(f), dif))
            dif = sorted(dif)
            values, counts = np.unique(dif, return_counts=True)
            lim = dif[-10]
            plt.figure()
            
            roots = []
            for v in f:
                if d[v] > lim:
                    roots.append(v)
                    
            mark = [f.index(i) for i in roots]
            plt.plot(range(len(f)), f, linewidth= 0.1) 
            plt.plot(range(len(f)), f,markevery=mark, ls="", marker="*", label="points", markersize=0.005)
            
            print(mark)
            df = []
            for m in mark:
                df.append(d[f[m]])
                
            print(df)
            
            plt.savefig('plot'+ str(cnt) +".png", dpi = 1200)
            cnt = cnt+1
        break

def add_x_Y(data_set, featuresets):
    converted_data = []
    for d in data_set:
        datapt = list([sum(x) for x in zip(d[featuresets[0]].tolist(), d[featuresets[1]].tolist())])
        converted_data.append(datapt)
        
    return converted_data
        
            
def plot_add(data_set, names):
    c = 1
    for d, text in zip(data_set, names):
        vals = np.array(d)
        mn = min(vals)
        vals = vals - mn
        mx = max(vals)
        vals = vals/mx
        
        plt.figure()
        plt.axvline(x = len(d)/4, color = 'r', linewidth= 0.1)
        plt.axvline(x = 2*(len(d)/4), color = 'r', linewidth= 0.1)
        plt.axvline(x = 3*(len(d)/4), color = 'r', linewidth= 0.1)
        fig = plt.figure()
        plt.plot(range(len(d)), vals, linewidth= 0.3) 
        fig.suptitle(text + '_x_y_' + str(c)+ ' ' + str(len(d)))
        plt.savefig(text + '_x_y_' + str(c)+ ' ' + str(len(d)) +".png", dpi = 1200)
        
        c = 1 if c == 5 else (c + 1)
    

original_data = original_vectors(obsdata, feature_list)
padding_end = padding_at_last(obsdata, feature_list)
padding_bw = padding_in_bw(obsdata, feature_list) 
interpolated = interpolating(obsdata)


for vector in original_data:
    vector = list(vector)
    ln = len(vector)
    n = int(ln/4)
    features = [vector[:ln][i:i + n] for i in range(0, ln, n)]
    print(len(features))
    cnt = 0 
    for f in features:
        
        dif = []
        for i in range(1, len(f)-1):
            dif.append(abs(((f[i-1] + f[i+1])/2) - f[i]))
                    
        dif.insert(0, dif[0])
        dif.insert(len(f), dif[-1])
        
        dif = list(dif)
        
        lim = sorted(dif)[-10]
        plt.figure()
        
        roots = []
        mark = []
        for h in range(n):
            if dif[h] > lim:
                roots.append(dif[h])
                mark.append(h)
                
        plt.plot(range(len(f)), f, linewidth= 0.1) 
        plt.plot(range(len(f)), f,markevery=mark, ls="", marker="*", label="points", markersize=0.005)
        print(mark)
            
        print(roots)
        
        #plt.savefig('plot'+ str(cnt) +".png", dpi = 1200)
        cnt = cnt+1
    break

#plot_all_diff(original_data, padding_end, padding_bw, interpolated, fig_names)