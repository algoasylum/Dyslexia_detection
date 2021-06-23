import numpy as np
import pandas as pd
import math

def conv(data_set):
    x = [sum(x) for x in zip(data_set['LX'].to_list(), data_set['RX'].tolist())]
    y = [sum(x) for x in zip(data_set['LY'].to_list(), data_set['RY'].tolist())]
    data = {'X':x, 'Y':y}
    
    return data 
    
def original_vector(data_set):
    data = []
    feature_list = ['X', 'Y']
    data_conv = conv(data_set)
    for f in feature_list:
        data += data_conv[f]

    return data

#Padding at the end
def get_padded(a,ml):
    a = np.pad(a, (0,abs(ml - len(a))), 'mean')
    
    return a 

def padding_at_last(data_set):
    data = []
    feature_list = ['X', 'Y']
    data_conv = conv(data_set)
    for f in feature_list:
        data += data_conv[f]
    
    return get_padded(data,3998)

def padding_in_bw(data_set):
    data = []
    feature_list = ['X', 'Y']
    data_conv = conv(data_set)
    for f in feature_list:
        data += get_padded(data_conv[f],1999).tolist()
        
    return data


def positions(secf, dif, fact):
    arr = []
    cn = 0
    for a in range(dif-1):
        cn = cn+secf + 1 if (a+1)%fact == 0 else cn+secf 
        arr.append(cn)
    
    return arr

def interpolating(data_set):
    feature_list = ['X', 'Y']
    data_conv = conv(data_set)
    
    combined_data = []
    ln = len(data_conv['X'])
    dif = 2000-ln
    sec = math.floor(ln/dif)

    pos_arr = positions(sec, dif, 1 if ln == 1499 else 2)

    for f in feature_list:
        data = []         
        curr = 0 
        for pos in range(1999):
            if curr < ln:
                if pos in pos_arr:
                    data.append((data_conv[f][curr]+data_conv[f][curr+1])/2)                        
                    data.append(data_conv[f][curr])
                    curr += 1
                else:
                    data.append(data_conv[f][curr])
                    curr += 1
        while len(data)<1999:
            last_val = data[-1]
            data.append(last_val)
            
        combined_data += data
    return combined_data

def exterpolation(data_set):  
    feature_list = ['X', 'Y']
    data_conv = conv(data_set)
    
    combined_data = []
    ln = len(data_conv['X'])
    dif = ln-998
    sec = math.floor(ln/dif)

    pos_arr = positions(sec, dif, 4 if ln == 1749 else 2)


    for f in feature_list:
        data = []  
        for pos in range(ln):
            y = 0
            if pos not in pos_arr:
                data.append(data_conv[f][pos])

        combined_data += data
        
    return combined_data