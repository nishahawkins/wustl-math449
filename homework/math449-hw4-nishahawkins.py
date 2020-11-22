import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse.extract import tril, triu
from numpy.linalg import norm

def getAuxMatrix(A):
    '''
    return D, L, U matrices for Jacobi or Gauss-Seidel
    D: array
    L, U: matrices
    '''
    # m = A.shape[0]
    D = csr_matrix.diagonal(A)
    L = -tril(A, k=-1)
    U = -triu(A, k=1)
    return D, L, U
def jacobiIteration(A, b, x0=None, tol=1e-13, numIter=100):
    '''
    Jacobi iteraiton:
    A: nxn matrix
    b: (n,) vector
    x0: initial guess
    numIter: total number of iteration
    tol: algorithm stops if ||x^{k+1} - x^{k}|| < tol
    return: x
    x: solution array such that x[i] = i-th iterate
    '''
    n = A.shape[0]
    x = np.zeros((numIter+1, n))
    if x0 is not None:
        x[0] = x0
    D, L, U = getAuxMatrix(A)
    for k in range(numIter):
        x[k+1] = ((L+U)@x[k])/D + b/D
        if norm(x[k+1] - x[k]) < tol:
            break
    
    return x[:k+1]
def simpleOptim(A, b):
    f = lambda x: 0.5 * np.transpose(x)@A@x-np.transpose(b)@x
    x_vec = jacobiIteration(A,b)
    x_min = x_vec[-1]
    f_min = f(x_min)
    return x_min, f_min