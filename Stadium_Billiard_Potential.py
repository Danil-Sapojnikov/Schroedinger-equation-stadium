# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:02:19 2025

@author: danil
"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

SAVE_FIGURE = False
FIGURE_NAME = 'Size_accuracy_plot'

L = 1 # height of stadium
NY = 100 # number of points along corresponding axis of box
NX = 2 * NY
delta = L/NY # size of discrete element

Num_Eig = 505
Num_bins = 50 #int(Num_Eig / 10)
SIGMA = 0 #-3 for high, 0 otherwise

epsilon = 10e-5
U_0 = 10e5

def plot_figure(num_eig, red_chi):

    #Plot 2 variable graph
    fig = plt.figure(figsize = (12,6))
    axes_main = fig.add_subplot(111)

    axes_main.plot(num_eig, red_chi)

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

    fig = plt.figure(figsize = (12,6))
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

def potential_plot():
    
    #temp potential plot
    contour_n = 0
    l = bunimovich_stadium(diag)
    l_sqr = np.reshape(l, (-1, NX))
    
    plot_contour(XX, YY, l_sqr, contour_n)
    
    return

def piecewise(x, y):

    sqr = abs(y)-L/2
    circ = (np.sqrt((abs(x)-L/2)**2 + abs(y)**2) - L/2)

    return np.where(abs(x)<=L/2, sqr, circ)

def bunimovich_stadium(diag):

    j = np.linspace(0,m-1,m)
    y = -L/2 + (np.trunc(j/NX)) * delta
    
    i = j - NX * np.trunc(j/NX)
    x = -L + i * delta
    
    return U_0 * (1 + np.tanh(piecewise(x,y)/epsilon))

def wigner(x):
    
    power = -np.pi / 4 * x**2
    
    return np.pi/2 * x * np.exp(power)

def poisson(x):

    return np.exp(-x)

def energy_diff(eigvals, func):
    
    eigvals_shifted = np.insert(eigvals, 0, 0)
    eigvals_shifted = np.delete(eigvals_shifted, -1)

    eigvals_diff = eigvals - eigvals_shifted
    mean = np.mean(eigvals_diff)
    
    eigvals_diff_scaled = eigvals_diff/mean
    
    counts, bins = np.histogram(eigvals_diff_scaled, bins=Num_bins)
    bin_width = bins[1] - bins[0]
    weights = counts / (np.sum(counts)*bin_width)
    plt.stairs(weights, bins)
    plt.plot(bins, func(bins))
    plt.show()
    plt.close()
    
    return counts, bins

def chi_sqr(observed, scaled_diff, func):

    scaled_diff = np.delete(scaled_diff, 0)
    deg_freedom = len(observed) - 1
    expected = func(scaled_diff)
    chi_squared = np.sum((observed - expected)**2 / expected)

    return chi_squared / deg_freedom

#discretise axis
m = NX*NY
diag = np.ones(m) #diagonals

#Define grid for contours
axis_x = np.linspace(-L, L - delta, NX)
axis_y = np.linspace(-L/2, L/2 - delta, NY)
XX, YY = np.meshgrid(axis_x, axis_y)

#potential_plot()

diag0 = diag*(-4) - bunimovich_stadium(diag) #main diagonal
diag1 = diag[0:-1] #second diagonal
diag1_bounding = diag1[NX-1::NX]
diag1_bounding[:] = 0 #prevents looping of wavefunction around box
diagk = diag[0:-NX] #k'th diagonal

#Form differential approximation matrix
M = scipy.sparse.diags([diag0, diag1, diag1, diagk, diagk], [0, 1, -1, NX, -NX], format='csc')

#Solve for Eigenvalues
#evals, evecs = scipy.sparse.linalg.eigsh(M, k=Num_Eig, which='LM')
evals, evecs = scipy.sparse.linalg.eigsh(M, sigma=SIGMA, k=Num_Eig, which='LM')

evals_sorted = -np.flip(np.sort(evals)) * (1/delta)**2 * np.pi**2

red_chi_sqr_array = np.empty(0)

for evals_num in range(100, 1001, 100):
    
    evals_slice = evals_sorted[:evals_num]

    weight, scaled_diff = energy_diff(evals_slice, poisson)

    red_chi_sqr = chi_sqr(weight, scaled_diff, poisson)
    red_chi_sqr_array = np.append(red_chi_sqr_array, red_chi_sqr)


amp_tot = np.empty((NY,NX))

"""
for contour_n in range(Num_Eig-180, Num_Eig):

    evecs_n = evecs[:,-(contour_n+1)]
    amp = np.reshape(evecs_n, (-1, NX))

    prob = amp**2

    #amp_tot += amp
    #prob = amp_tot**2

    plot_contour(XX, YY, prob, contour_n)
"""