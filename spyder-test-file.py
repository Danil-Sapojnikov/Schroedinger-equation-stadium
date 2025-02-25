import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
#import matplotlib.pylab as plt

SAVE_FIGURE = False
FIGURE_NAME = 'Size_accuracy_plot'


def plot_figure(size, ediffSet):

    #Plot 2 variable graph
    fig = plt.figure(figsize = (9,12))
    axes_main = fig.add_subplot(111)
    
    for i in range(0,10):
        axes_main.plot(size, ediffSet[:,i], color='b')

    axes_main.set_title('Size - Accuracy Plot',
                        fontsize = 14, fontweight=550)
    axes_main.grid()
    axes_main.set_xlabel('n x n grid', fontsize = 11)
    axes_main.set_ylabel('E_analytical/E_calculated', fontsize = 11)
    #axes_main.legend(fontsize = 13)

    #Saves figure
    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, transparent = True)
    plt.show()

    plt.close()

    return

size = np.empty((0))
ediffSet = np.empty((0,10))
nStates = np.array((2,5,5,8,10,10,13,13,18,25))

for dim in range(10, 101, 1):
    n = dim**2
    d = np.ones(n) #diagonals
    b = np.zeros(n) #RHS
    d0 = d*(-4) #main diagonal
    d1 = d[0:-1] #second diagonal
    d5 = d[0:-dim] #l'th diagonal

    M = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, dim, -dim], format='csc')

    evals, evecs = scipy.sparse.linalg.eigs(M, k=10, which='SM')
    
    size = np.hstack((size,dim))
    
    analyticVal = -np.pi**2/dim**2 * nStates
    
    ediff = analyticVal/evals
    
    ediffSet = np.vstack((ediffSet, ediff))

plot_figure(size, ediffSet)
