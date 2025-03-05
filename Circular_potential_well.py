# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:37:04 2025

@author: danil
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

SAVE_FIGURE = False
FIGURE_NAME = 'Size_accuracy_plot'

L = 1 # length of box
n = 200 # number of points along each axis of box
delta = L/n # size of discrete element

epsilon = 10e-3
U_0 = 10e3

def plot_figure(size, ediffSet):

    #Plot 2 variable graph
    fig = plt.figure(figsize = (12,6))
    axes_main = fig.add_subplot(111)
    
    for i in range(0,len(ediffSet[0,:])):
        axes_main.plot(size, ediffSet[:,i], label=str(i))

    axes_main.set_title('Size - Accuracy Plot',
                        fontsize = 14, fontweight=550)
    axes_main.grid()
    axes_main.set_xlabel('n x n grid', fontsize = 11)
    axes_main.set_ylabel('E_analytical/E_calculated', fontsize = 11)
    axes_main.legend(fontsize = 13)

    #Saves figure
    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, transparent = True)
    plt.show()

    plt.close()

    return

def circ_potential(diag):
    
    j = np.linspace(0,m-1,m)
    y = -L/2 + (np.trunc(j/n)) * delta
    
    i = j - n * np.trunc(j/n)
    x = -L/2 + i * delta
    
    r = np.sqrt(x**2+y**2)
    
    return U_0 * np.tanh((r - L/2)/epsilon)
    
#discretise axis
m = n**2
diag = np.ones(m) #diagonals
 

#x = + i*delta
#y = + j*delta

diag0 = diag*(-4) - circ_potential(diag) #main diagonal
diag1 = diag[0:-1] #second diagonal
diag1_bounding = diag1[n-1::n]
diag1_bounding[:] = 0 #prevents looping of wavefunction around box
diagk = diag[0:-n] #k'th diagonal

#Form differential approximation matrix
M = scipy.sparse.diags([diag0, diag1, diag1, diagk, diagk], [0, 1, -1, n, -n], format='csc')

#Solve for Eigenvalues
#evals, evecs = scipy.sparse.linalg.eigs(M, k=10, which='SM')
evals, evecs = scipy.sparse.linalg.eigsh(M, sigma=0, k=100, which='LM')

evals_sorted = -np.flip(np.sort(evals)) * (1/delta)**2 
