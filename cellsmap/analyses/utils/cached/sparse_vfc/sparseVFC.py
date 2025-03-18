import numpy as np
import numpy.random as rnd
import scipy as sc

from cellsmap.analyses.utils.kernels import *

global kernel_func_dict
kernel_func_dict = {'rbf':rbf_kernel,'laplace':laplace_kernel,'linear':linear_kernel,'polynomial':polynomial_kernel}

def sparseVFC_alg(X,Y,kernel_func=rbf_kernel,kernel_params={'beta':0.1},lam=3,M=16,gamma=0.9,a=10,theta=0.75,minP=1e-5,tol_ecr=1e-5,silent=True,MaxIter=500,ctrl_pts=[],seed=None):
    '''
    Implementation of sparseVFC algorithm for vector field regression with RBF kernel (Ma et al. 2013). Adapted from the
    original R code, found at https://github.com/Sciurus365/SparseVFC.

    TO DO: add documentation explaining the algorithm and its parameters
    '''
    N = X.shape[0] # number of training data points
    D = X.shape[1] # dimension of the state space

    if np.ndim(Y) == 1: Y = Y[:,None]

    # Construct kernel matrix K over control points
    if len(ctrl_pts) == 0: # if not supplied, choose M random points from X
        if seed is not None: rng = rnd.default_rng(seed) # set random seed for reproducibility
        else: rng = rnd.default_rng()
        temp_X = np.unique(X,axis=0)
        idx = rng.choice(range(temp_X.shape[0]),temp_X.shape[0],replace=False) # random permutation of indices
        ctrl_idx = idx[0:min(M, temp_X.shape[0])] # indices of control points: first M indices or all indices if M >= number of unique X values
        ctrl_pts = temp_X[ctrl_idx,:] # control points
    if not silent: print("Precomputing kernel matrices...\n")
    K = kernel_func(ctrl_pts,ctrl_pts,**kernel_params) # kernel matrix over control points
    U = kernel_func(X,ctrl_pts,**kernel_params) # kernel matrix between X and control points
    M = ctrl_pts.shape[0]

    # Initialization
    V = np.zeros((N,D))
    iter = 0
    ecr = 1 
    E = 1
    sigma2 = np.sum((Y - V)**2) / (N * D)

    illCond = False

    if not silent: print("Start mismatch removal...\n")
    while iter < MaxIter and ecr > tol_ecr:
        # E-step.
        E_old = E
        P,E = get_P(Y, V, sigma2, gamma, a)
        ecr = abs((E - E_old) / E) # energy change rate: used as measure of convergence
        if not silent: print("iterate %d, gamma: %f, energy change rate: %f, sigma2=%f\n" % (iter+1, gamma, ecr, sigma2))

        # M-step. Solve linear system for C (matrix of coefficient vectors that give us the fit function).
        P = np.maximum(P, minP) # take pairwise maximum between elements of P and minP

        if not silent and np.linalg.cond(np.multiply(U.T,P)@U + lam*sigma2*K) > 1e10 and illCond==False: 
            print("Warning: Ill conditioned problem (condition number > 1e10)","\n")
            print("lambda:"+f"{lam:.3f}"+", beta:"+f"{kernel_params['beta']:.3f}"+", M:"+str(M),"\n")
            illCond = True

        C = np.linalg.pinv(np.multiply(U.T,P)@U + lam*sigma2*K)@(np.multiply(U.T,P)@Y)
        
        # Update V and sigma^2
        V = U @ C
        Sp = np.sum(P)
        sigma2 = np.sum(P*np.sum((Y - V)**2,axis=1))/(Sp*D)

        # Update gamma
        numcorr = np.sum(P[P > theta]) # number of inlier points after this iteration
        gamma = numcorr/N # gamma is the fraction of inlier points
        if gamma > 0.95: gamma = 0.95
        elif gamma < 0.05: gamma = 0.05

        iter += 1

    if not silent: print("Removing outliers succesfully completed.")

    return ctrl_pts, V, C, P, np.where(P > theta)[0], sigma2

# Estimate the posterior probability and part of the energy
def get_P(Y, V, sigma2, gamma, a):
  D = Y.shape[1]
  temp1 = np.exp(-np.sum((Y - V)**2,axis=1)/(2*sigma2))
  temp2 = (2*np.pi*sigma2)**(D/2)*(1-gamma)/(gamma*a)
  P = temp1 / (temp1 + temp2) # storing P as vector, not diagonal matrix
  E = np.sum(P*np.sum((Y - V)**2,axis=1)/(2*sigma2)) + np.sum(P)*np.log(sigma2)*D/2
  return P,E

class SparseVFC:
    def __init__(self,kernel='rbf',beta=0.1,coeff=1,degree=2,lam=3,M=16,gamma=0.9,a=10,theta=0.75,minP=1e-5,tol_ecr=1e-5,silent=True,MaxIter=500,seed=None):
        self.kernel = kernel
        self.beta = beta
        self.coeff = coeff
        self.degree = degree
        self.lam = lam
        self.M = M
        self.gamma = gamma
        self.a = a
        self.theta = theta
        self.minP = minP
        self.tol_ecr = tol_ecr
        self.silent = silent
        self.MaxIter = MaxIter
        self.seed = seed

    def check_XY(self,X,Y): 
        # for checking input data, estimator class needs to raise certain value errors to be sklearn compatible
        nanstrlist = ['nan','NaN','NAN']
        infstrlist = ['inf','INF','Inf']
        if X.__class__== list:
            if any([x.__class__==str and x in nanstrlist for x in X]): 
                raise ValueError('Input contains NaN')
            if any([x.__class__==str and x in infstrlist for x in X]): 
                raise ValueError('Input contains inf')
            np.array(X)
        if Y.__class__== list:
            if any([y.__class__==str and y in nanstrlist for y in Y]): 
                raise ValueError('Input contains NaN') 
            if any([y.__class__==str and y in infstrlist for y in Y]): 
                raise ValueError('Input contains inf') 
            np.array(Y)

        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(Y)):
            raise ValueError('Complex data not supported')
        
        if X.ndim == 0 or X.size == 0:
            raise ValueError("0 feature(s) (shape=(%d, 0)) while a minimum of %d is required." % (X.shape[0],1))
        if Y.ndim == 0 or Y.size == 0:
            raise ValueError('Empty target data')
        
        if sc.sparse.issparse(X): 
            raise ValueError("Sparse data are not supported")
        
        X = X.astype(float)
        Y = Y.astype(float)

        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            raise ValueError('Input contains NaN')
        if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
            raise ValueError('Input contains inf')

        if X.ndim == 1: 
            raise ValueError("1D array supplied, must be 2D")

        return X,Y
    
    def check_X(self,X): 
        # checking input for self.predict (neccessary for sklearn compatibility)
        nanstrlist = ['nan','NaN','NAN']
        infstrlist = ['inf','INF','Inf']
        if X.__class__== list:
            if any([x.__class__==str and x in nanstrlist for x in X]): 
                raise ValueError('Input contains NaN')
            if any([x.__class__==str and x in infstrlist for x in X]): 
                raise ValueError('Input contains inf')
            X = np.array(X)

        if np.any(np.iscomplex(X)):
            raise ValueError('Complex data not supported')
        
        if X.ndim == 0 or X.size == 0:
            raise ValueError("0 feature(s) (shape=(%d, 0)) while a minimum of %d is required." % (X.shape[0],1))

        X = X.astype(float)

        if np.any(np.isnan(X)):
            raise ValueError('Input contains NaN')
        if np.any(np.isinf(X)):
            raise ValueError('Input contains inf')
        
        if X.ndim == 1 and self.n_features_in_ != 1:
            raise ValueError("Reshape your data")
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("Reshape your data")

        return X
    
    def fit(self,X,Y):
        X,Y  = self.check_XY(X,Y) # check input data, make sure no ValueError is raised (neccessary for sklearn compatibility)
        self.X_train_ = X
        self.Y_train_ = Y
        self.n_features_in_ = X.shape[1]

        # self.kernel_func_ = kernel_func_dict[self.kernel] # set kernel function based on input kernel, pass in kernel parameters as well
        self.kernel_func_ = kernel_func_dict[self.kernel]

        self._get_kernel_params() # set kernel parameter dictionary based on input kernel

        if hasattr(self,'ctrl_pts_'):
            self.ctrl_pts_, self.V_, self.C_, self.P_, self.inliers_, self.sigma2_ = sparseVFC_alg(X,Y,kernel_func=self.kernel_func_,kernel_params=self.kernel_params_,
                                                                                                   lam=self.lam,M=self.M,gamma=self.gamma,a=self.a,
                                                                                                   theta=self.theta,minP=self.minP,tol_ecr=self.tol_ecr,
                                                                                                   silent=self.silent,MaxIter=self.MaxIter,ctrl_pts=self.ctrl_pts_,seed=self.seed)
        else: 
            self.ctrl_pts_, self.V_, self.C_, self.P_, self.inliers_, self.sigma2_ = sparseVFC_alg(X,Y,kernel_func=self.kernel_func_,kernel_params=self.kernel_params_,
                                                                                                   lam=self.lam,M=self.M,gamma=self.gamma,a=self.a,
                                                                                                   theta=self.theta,minP=self.minP,tol_ecr=self.tol_ecr,
                                                                                                   silent=self.silent,MaxIter=self.MaxIter)
        return self

    def predict(self,X):
        X = self.check_X(X) # check input data, make sure no ValueError is raised (neccessary for sklearn compatibility)
        
        return self.kernel_func_(X,self.ctrl_pts_,**self.kernel_params_)@self.C_

    def score(self,X,Y):
        X,Y = self.check_XY(X,Y) # check input data, make sure no ValueError is raised
        
        RSS = np.sum(np.linalg.norm(Y-self.predict(X),axis=-1)**2) # residual sum of squares
        TSS = np.sum(np.linalg.norm(np.mean(Y,axis=0)-Y,axis=-1)**2) # total sum of squares
        return 1 - RSS/TSS # coefficient of determination R^2
    
    def project2D(self,W):
        '''
        Given trained sparse VFC model self and projection matrix W (e.g., truncated PCA basis),
        sets projected coefficient matrix self.C_proj = self.C_ @ W.
        '''
        self.proj_ = W # save projection matrix
        self.C_proj_ = self.C_ @ W
        self.ctrl_pts_proj_ = self.ctrl_pts_ @ W # project control points to 2D
        return self

    def predict_2D(self,X_proj):
        '''
        Returns the projected vector field at (n x 2) array X_proj (not mesh grid).
        '''
        return self.kernel_func_(X_proj,self.ctrl_pts_proj_,**self.kernel_params_)@self.C_proj_
    
    def predict_2D_mesh(self,mesh_grid):
        '''
        Returns the projected vector field V over 2D mesh grid mesh_grid [U1,U2].
        '''
        n_1 = mesh_grid[0].shape[0]
        n_2 = mesh_grid[0].shape[1]
        V = np.zeros((n_1,n_2,2))
        for i in range(n_1):
            V[i,:,:] = self.kernel_func_(np.vstack((mesh_grid[0][i,:],mesh_grid[1][i,:])).T,self.ctrl_pts_proj_,**self.kernel_params_)@self.C_proj_
        return V

    def _get_kernel_params(self):
        if self.kernel == 'rbf' or self.kernel == 'laplace':
            self.kernel_params_ = {'beta':self.beta}
        elif self.kernel == 'polynomial':
            self.kernel_params_ = {'beta':self.beta,'coeff':self.coeff,'degree':self.degree}
        elif self.kernel == 'sigmoid':
            self.kernel_params_ = {'beta':self.beta,'coeff':self.coeff}
        else: # linear kernel, no parameters
            self.kernel_params_ = {}
        return self

    def get_params(self,deep=True):
        return {'kernel':self.kernel,
                'beta':self.beta,
                'coeff':self.coeff,
                'degree':self.degree,
                'lam':self.lam,
                'M':self.M,
                'gamma':self.gamma,
                'a':self.a,
                'theta':self.theta,
                'minP':self.minP,
                'tol_ecr':self.tol_ecr,
                'silent':self.silent,
                'MaxIter':self.MaxIter,
                'seed':self.seed}

    def set_params(self,**parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    