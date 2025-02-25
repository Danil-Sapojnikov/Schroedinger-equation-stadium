import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pylab as plt

yaxis = np.empty((0,1))

n = 100
d = np.ones(n) #diagonals
b = np.zeros(n) #RHS
d0 = d*(4) #main diagonal
d1 = d[0:-1] #second diagonal
d5 = d[0:-5]

M = scipy.sparse.diags([d0, d1, d1, d5, d5], [0, 1, -1, n, -n], format='csc')

#M[0,:], M[24,:], M[:,0], M[:,24] = 1000, 1000, 1000, 1000

eig = scipy.sparse.linalg.eigs(M)

evals = eig[0]
evecs = eig[1]
