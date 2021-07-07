import numpy as np
import pandas as pd
import math
from Data_load import average_l_r 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note: 'data_set' is referenced as input to most functions
# data_set: data of a single candidate as dataframe(structure: LX    LY    RX    RY)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def original_vector(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#appends the 'X' and 'Y' values of the dataset one after the other 

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values
#  data = xxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyyyyy
#        |________________________||________________________|       
#                    x                        y

#OUTPUT:
#     data : Appended avg of x and y values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += data_conv[f]

    return data

def get_padded(a,ml):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Adds 0 padding at the end to make vector length equal to 'ml'
#INPUT:
#     a = vector to be padded
#     ml = target length after padding

#   a     =   ////////////////////////

#padded a =   ////////////////////////0000000
#             |<-------------ml------------->|
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    a = np.pad(a, (0,abs(ml - len(a))), 'mean')
    
    return a 

def padding_at_last(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Combines the data vectors (X, Y) one after the other into a single vector. Added 0s towards the end to match the length of the longest vector if just x and y values are combined without any padding. 
#In our dataset the longest vector is of length 3998.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

##  data = xxxxxxxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyyyyyy0000000000000000
#          |_______________________||_______________________||______________|       
#                     x                        y                    0s
#          |<-------------------length of longest vector------------------->|

#OUTPUT:
#     data: x, y combined and padded with 0s
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += data_conv[f]
    
    return get_padded(data,3998)

def padding_in_bw(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Pads each feature (X and Y) separately to match the longest length of a single fearture vector. In our case it's 1999. Then appends them together.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

# padded x =  xxxxxxxxxxxxxxxxxxxxxxxxx00000000
#             |_______________________||______|
#                         x               0s          

# padded y =  yyyyyyyyyyyyyyyyyyyyyyyyy00000000
#             |_______________________||______|
#                         y               0s    

#             |<----------------------------->|
#          length of longest single fearture vector

# data =  xxxxxxxxxxxxxxxxxxxxxxxxx00000000yyyyyyyyyyyyyyyyyyyyyyyyy00000000
#         |_______________________||______||_______________________||______|
#                     x               0s               y               0s     

#OUTPUT:
#     data: combined padded x and padded y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    data = []
    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    for f in feature_list:
        data += get_padded(data_conv[f],1999).tolist()
        
    return data


def positions(secf, dif, fact):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Gives indexes of equally spaced positions in a vector to add or remove values to match a certain length. In our case it's 1999.
#INPUT: 
#     secf:  Length of each equal space
#     dif:  Number of positions to be added/removed
#     fact:  

#OUTPUT:
#     arr: vector containing 'dif' number of equally spaced positions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    arr = []
    cn = 0
    for a in range(dif-1):
        cn = cn+secf + 1 if (a+1)%fact == 0 else cn+secf 
        arr.append(cn)
    
    return arr

def interpolating(data_set):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Adds data points at regular intervals by taking the average of the adjacent values to match the longest length of a single fearture vector. In our case it's 1999.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

# interpolated x =  xxxxxXxxxxxXxxxxxXxxxxxXxxxxx
#                  |_____^_____^_____^_____^___|
#           x with values inserted at regualar intervals                 

# interpolated y =  yyyyyYyyyyyYyyyyyYyyyyyYyyyyy
#                  |_____^_____^_____^_____^___|
#           y with values inserted at regualar intervals    

#             |<----------------------------->|
#          length of longest single fearture vector

# combined_data =   xxxxxXxxxxxXxxxxxXxxxxxXxxxxxyyyyyYyyyyyYyyyyyYyyyyyYyyyyy
#                  |____________________________||___________________________|
#                          interpolated x               interpolated y     

#OUTPUT:
#     combined_data: Appended interpolated x and inpterpolated y together
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Removes data points at regular intervals to match the shortest length of a single fearture vector. In our case it's 999.

#  x = xxxxxxxxxxxxxxxxxxxxxxxxx, y = yyyyyyyyyyyyyyyyyyyyyyyyy
#          avg of x values             avg of x values

#                       x   x   x   x
# exterpolated x =  xxxxxxxxxxxxxxxxxxxx
#                  |___^___^___^___^___|
#           x with values removed at regualar intervals                 

#                       y   y   y   y
# exterpolated y =  yyyyyyyyyyyyyyyyyyyy
#                  |___^___^___^___^___|
#           y with values removed at regualar intervals    

#                  |<----------------->|
#          length of shortest single fearture vector

# combined_data =   xxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyy
#                  |___________________||__________________|
#                      exterpolated x      exterpolated y     

#OUTPUT:
#     combined_data: Appended exterpolated x and exterpolated y together
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    feature_list = ['X', 'Y']
    data_conv = average_l_r(data_set)
    
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