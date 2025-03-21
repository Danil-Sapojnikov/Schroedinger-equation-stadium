# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:00:58 2025

@author: danil
"""

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

def plot_contour(XX, YY, prob, contour_N):

    fig = plt.figure(figsize = (6,6))
    axes_main = fig.add_subplot(111)

    axes_main.set_title(f'Wavefuntion n = {contour_N}',
                        fontsize = 10, fontweight=550)
    axes_main.set_xlabel('x', fontsize = 10)
    axes_main.set_ylabel('y', fontsize = 10)

    axes_main.contourf(XX, YY, prob, levels = 50)
    
    #Saves figure
    if SAVE_FIGURE:
        plt.savefig(FIGURE_NAME, transparent = True)
    plt.show()

    plt.close()

    return

#nStates = np.array((2,5,5,8,10,10,13,13,17,17))
N = 100
L = 1
delta = L/N

Num_Eig = 100
#contour_n = 0

#discretise axis
m = N**2
diag = np.ones(m) #diagonals
diag0 = diag*(-4) #main diagonal
diag1 = diag[0:-1] #second diagonal
diag1_bounding = diag1[N-1::N]
diag1_bounding[:] = 0 #prevents looping of wavefunction around box
diagk = diag[0:-N] #k'th diagonal

#Form differential approximation matrix
M = scipy.sparse.diags([diag0, diag1, diag1, diagk, diagk], [0, 1, -1, N, -N], format='csc')

#Solve for Eigenvalues
#evals, evecs = scipy.sparse.linalg.eigs(M, k=10, which='SM')
evals, evecs = scipy.sparse.linalg.eigsh(M, sigma=0, k=Num_Eig, which='LM')

#evals_sorted = np.flip(np.sort(evals))
evals_sorted = -np.flip(np.sort(evals)) * (N)**2 / np.pi**2

#Contour plot of wavefunction
axis_x = np.linspace(-L/2, L/2 - delta, N)
axis_y = axis_x
XX, YY = np.meshgrid(axis_x, axis_y)

for contour_n in range(0, Num_Eig):

    evecs_n = evecs[:,-(contour_n+1)]
    amp = np.reshape(evecs_n, (-1, N))
    prob = amp**2

    plot_contour(XX, YY, prob, contour_n)
