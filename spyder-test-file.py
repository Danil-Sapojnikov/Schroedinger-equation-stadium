import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pylab as plt

yaxis = np.empty((0,1))

dim = 10
n = dim**2
d = np.ones(n) #diagonals
b = np.zeros(n) #RHS
d0 = d*(-4) #main diagonal
d1 = d[0:-1] #second diagonal
d5 = d[0:-dim] #l'th diagonal

M = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, dim, -dim], format='csc')

eig = scipy.sparse.linalg.eigs(M, k=n-2)

evals = eig[0]
evecs = eig[1]


