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

size = np.empty((0))
ediffSet = np.empty((0,10))
nStates = np.array((2,5,5,8,10,10,13,13,17,17))
#nStates = np.array((1,4,5,5,8,8,9,13,13,16))
#nStates = np.array((1,1,2,2,2,4,4,5,5,8))
#nStates = np.array((2,5,5,8,13,13,17,17,18,20))



for n in range(10, 101, 1):
    
    #discretise axis
    m = n**2
    diag = np.ones(m) #diagonals
    diag0 = diag*(-4) #main diagonal
    diag1 = diag[0:-1] #second diagonal
    diag1_bounding = diag1[n-1::n]
    diag1_bounding[:] = 0 #prevents looping of wavefunction around box
    diagk = diag[0:-n] #k'th diagonal

    #Form differential approximation matrix
    M = scipy.sparse.diags([diag0, diag1, diag1, diagk, diagk], [0, 1, -1, n, -n], format='csc')

    #Solve for Eigenvalues
    #evals, evecs = scipy.sparse.linalg.eigs(M, k=10, which='SM')
    evals, evecs = scipy.sparse.linalg.eigsh(M, sigma=0, k=10, which='LM',tol = 1e-5)
    
    #evals_sorted = np.flip(np.sort(evals))
    evals_sorted = -np.flip(np.sort(evals)) * (n)**2 / np.pi**2
    
    #Stack Eigenvalues for comparison graph
    size = np.hstack((size,n))
    
    #analyticVal = np.pi**2 * nStates
    #analyticVal = np.pi**2/dim**2 * nStates
    
    #ediff = evals_sorted/nStates
    #ediff = evals_sorted/analyticVal
    #ediff = analyticVal/evals_sorted
    ediff = evals_sorted
    
    ediffSet = np.vstack((ediffSet, ediff))

plot_figure(size, ediffSet)
