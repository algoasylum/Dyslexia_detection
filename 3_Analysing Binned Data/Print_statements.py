import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import euclidean as eu

def plot_entire_candidate(category, num, data):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To plot the different features of a candidate in a 2x2 grid.
#INPUT: 
#     category: Dyslexic or Control, 0 for Control and 1 for Dyslexic
#     num: serial number of the entry which is to be printed
#     data: entire dataset

#Structure of plot: 
# LX  LY
# RX  RY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=2)
    gs.update(wspace = 0.2, hspace = 0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(data[category][num]['LX'], linewidth = 0.7)
    ax0.set_title('LX')

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(data[category][num]['RX'], linewidth = 0.7)
    ax1.set_title('RX')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(data[category][num]['LY'], linewidth = 0.7)
    ax2.set_title('LY')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(data[category][num]['RY'], linewidth = 0.7)
    ax3.set_title('RY')


def plot_left_right(category, num, data, axis):
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is to plot left and right eye readings of one axis for one candidate. This is to mainly check the overlap betweeen the 2 sides. 
#INPUT: 
#     category: Dyslexic or Control, 0 for Control and 1 for Dyslexic
#     num: serial number of the entry which is to be printed
#     data: entire dataset
#     axis: Which axis to be printed - x or y
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.plot(data[category][num]['R'+axis], 'r', label = 'Right', alpha=1)
    plt.plot(data[category][num]['L'+axis], 'y', label = 'Left', alpha=0.7)
    plt.legend()
    
def return_sq_im(set1, set2):    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculated the distances of element of the first set to each in the second set and represented them with shades of grey between white(max distance) and black(no distance).
#Each intersection/pixel represents the distance between the corresponding vectors.
#Eg. The grade of grey of the pixel at (4,1) will represent the distance between the 4th element of the first set to the 1st element of the second set
#INPUT:
#     set1: First set of vectors
#     set2: Second set of vectors
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    matrix = np.empty([len(set1),len(set2)])    
    for a in range(len(set1)):
        for b in range(len(set2)):
            m = set1[a]
            n = set2[b]

            dis = eu([m], [n])
            matrix[a][b] = dis 

    mx = np.amax(matrix)
    img = matrix/mx  
    img = img*256
    
    plt.imshow(img, cmap='gray', vmin=0, vmax=256)
    plt.show()