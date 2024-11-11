import numpy as np

def rbf_kernel(X,Y,beta=0.1):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    m = Y.shape[0]
    n = X.shape[0]
 
    K = np.zeros((n,m))
    for i in range(n):
        K[i,:] = np.exp(-beta*np.linalg.norm(X[i,:]-Y,axis=-1)**2)

    return K

def laplace_kernel(X,Y,beta=0.1):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    m = Y.shape[0]
    n = X.shape[0]
 
    K = np.zeros((n,m))
    for i in range(n):
        K[i,:] = np.exp(-beta*np.linalg.norm(X[i,:]-Y,axis=-1,ord=1))

    return K

def linear_kernel(X,Y):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    m = Y.shape[0]
    n = X.shape[0]
 
    K = np.zeros((n,m))
    for i in range(n):
        K[i,:] = Y@X[i,:]

    return K

def polynomial_kernel(X,Y,beta=0.1,coeff=1,degree=2):
    if X.ndim == 1:
        X = X.reshape(1,-1)
    m = Y.shape[0]
    n = X.shape[0]
 
    K = np.zeros((n,m))
    for i in range(n):
        K[i,:] = (beta*Y@X[i,:]+coeff)**degree

    return K

def matrix_kernel(kernel_func,params,out_dim,X,Y,B=None):
    if B is None: 
        B = np.eye(out_dim)
    
    if X.ndim == 1:
        X = X.reshape(1,-1)
    if Y.ndim == 1:
        Y = Y.reshape(1,-1)
    m = Y.shape[0]
    n = X.shape[0]
 
    K = np.zeros((n,m))
    for i in range(n):
        K[i,:] = kernel_func(X,Y,**params)

    return np.kron(K,B)