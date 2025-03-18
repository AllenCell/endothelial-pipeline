import numpy as np
import numpy.random as rnd
import scipy as sc

from cellsmap.analyses.utils.kernels import *

global kernel_func_dict
kernel_func_dict = {'rbf':rbf_kernel,'laplace':laplace_kernel,'linear':linear_kernel,'polynomial':polynomial_kernel}

def kernel_reg(X,Y,kernel_func=rbf_kernel,kernel_params={'beta':0.1},lam=3,weights=None,matrix_kernel=False,B=None):
    '''
    Kernel regression function for L2 regularized least squares loss function.
    '''
    N = X.shape[0] # number of training data points

    if np.ndim(Y) == 1: 
        Y = Y[:,None]

    # Construct kernel matrix K over data points
    if matrix_kernel:
        D = Y.shape[1] # number of output features (dimension of the state space)
        K = matrix_kernel(kernel_func,kernel_params,D,X,X,B=B)
    else:
        K = kernel_func(X,X,**kernel_params) # kernel matrix over input data points

    # Weight matrix P
    if weights is None:
        Pinv = np.eye(N)
    else:
        Pinv = np.diag(1/weights)
    
    if matrix_kernel:
        Pinv = np.kron(Pinv,np.eye(D))

    return np.linalg.pinv(K+lam*Pinv)@Y


class KernelRegression:
    def __init__(self,kernel='rbf',beta=0.1,coeff=1,degree=2,lam=3,weights=None,matrix_kernel=False,B=None):
        self.kernel = kernel
        self.beta = beta
        self.coeff = coeff
        self.degree = degree
        self.lam = lam
        self.weights = weights
        self.matrix_kernel = matrix_kernel
        self.B = B

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
    
    def check_u(self,u):
        # check control input
        if u.__class__ in [int,float,complex]:
            return u
        elif u.__class__ == list:
            if any([x.__class__==str and x in ['nan','NaN','NAN'] for x in u]): 
                raise ValueError('Input contains NaN')
            if any([x.__class__==str and x in ['inf','INF','Inf'] for x in u]): 
                raise ValueError('Input contains inf')
            u = np.array(u)
        if np.any(np.iscomplex(u)):
            raise ValueError('Complex data not supported')
        if np.any(np.isnan(u)):
            raise ValueError('Input contains NaN')
        if np.any(np.isinf(u)):
            raise ValueError('Input contains inf')
        if u.ndim == 0 or u.size == 0:
            raise ValueError("0 feature(s) (shape=(%d, 0)) while a minimum of %d is required." % (u.shape[0],1))
        if u.ndim == 1 and self.n_features_in_ != 1:
            raise ValueError("Reshape your data")
        elif u.shape[1] != self.n_features_in_:
            raise ValueError("Reshape your data")
    
    def fit(self,X,Y,u=None):
        X,Y  = self.check_XY(X,Y) # check input data, make sure no ValueError is raised (neccessary for sklearn compatibility)
        self.X_train_ = X
        self.Y_train_ = Y
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = Y.shape[1]

        if self.matrix_kernel:
            self.kernel_func_ = lambda X,Y,params: matrix_kernel(kernel_func_dict[self.kernel],
                                                                 params,self.n_features_out_,
                                                                 X,Y,B=self.B)
        self.kernel_func_ = kernel_func_dict[self.kernel]

        self._get_kernel_params() # set kernel parameter dictionary based on input kernel

        if u is not None:
            # write these checks as a function
            if u.__class__ in [int,float,complex]:
                if isinstance(X,np.ndarray):
                    u = u*np.ones((X.shape[0],1))
                else:
                    u = np.array([u])  
            elif u.shape[0] != X.shape[0] and u.T.shape[0] != X.shape[0]:
                raise ValueError('Control input must have same number of data points as input data')          
            elif len(u.shape) == 1:
                u = u[:,None]
            self.u_train_ = u
            X_aug = np.hstack((X,u)) # augment input data with control input
            self.X_train_aug_ = X_aug
            self.C_ = kernel_reg(X_aug,Y,self.kernel_func_,
                             self.kernel_params_,self.lam,
                             self.weights,self.matrix_kernel,self.B) # fit kernel regression model
        else:
            self.C_ = kernel_reg(X,Y,self.kernel_func_,
                             self.kernel_params_,self.lam,
                             self.weights,self.matrix_kernel,self.B) # fit kernel regression model
        return self

    def predict(self,X,u=None):
        X = self.check_X(X) # check input data, make sure no ValueError is raised (neccessary for sklearn compatibility)
        
        if u is not None:
            if not hasattr(self,'u_train_'):
                raise ValueError('Model was not trained with control input')    
            if u.__class__ in [int,float,np.float64,complex]:
                if isinstance(X,np.ndarray):
                    u = u*np.ones((X.shape[0],1))
                else:
                    u = np.array([u])  
            elif u.shape[0] != X.shape[0] and u.T.shape[0] != X.shape[0]:
                raise ValueError('Control input must have same number of data points as input data')          
            elif len(u.shape) == 1:
                u = u[:,None]
            X_aug = np.hstack((X,u))
            return self.kernel_func_(X_aug,self.X_train_aug_,**self.kernel_params_)@self.C_
        else:
            if hasattr(self,'u_train_'):
                raise ValueError('Model was trained with control input, predict requires control input u.')
            return self.kernel_func_(X,self.X_train_,**self.kernel_params_)@self.C_

    def score(self,X,Y,u=None):
        X,Y = self.check_XY(X,Y) # check input data, make sure no ValueError is raised
        
        RSS = np.sum(np.linalg.norm(Y-self.predict(X,u),axis=-1)**2) # residual sum of squares
        TSS = np.sum(np.linalg.norm(np.mean(Y,axis=0)-Y,axis=-1)**2) # total sum of squares
        return 1 - RSS/TSS # coefficient of determination R^2
    
    def project2D(self,W):
        '''
        Given trained sparse VFC model self and projection matrix W (e.g., truncated PCA basis),
        sets projected coefficient matrix self.C_proj = self.C_ @ W.
        '''
        self.proj_ = W # save projection matrix
        self.C_proj_ = self.C_ @ W
        if self.B is not None:
            self.B_proj_ = self.B @ W
        self.X_train_proj_ = self.X_train_ @ W # project control points to 2D
        if hasattr(self,'u_train_'):
            self.X_train_aug_proj_ = np.hstack((self.X_train_proj_,self.u_train_)) # augment projected control points with control input
        return self

    def predict_2D(self,X_proj,u=None):
        '''
        Returns the projected vector field at (n x 2) array X_proj (not mesh grid).
        '''
        if u is not None:
            if not hasattr(self,'u_train_'):
                raise ValueError('Model was not trained with control input')    
            X_aug = np.hstack((X_proj,u))
            if self.matrix_kernel:
                return matrix_kernel(self.kernel_func_,self.kernel_params_,2,X_aug,self.X_train_aug_proj_,B=self.B_proj_)@self.C_proj_
            else:
                return self.kernel_func_(X_aug,self.X_train_aug_proj_,**self.kernel_params_)@self.C_proj_
        else:
            if hasattr(self,'u_train_'):
                raise ValueError('Model was trained with control input, predict requires control input u.')    
            if self.matrix_kernel:
                return matrix_kernel(self.kernel_func_,self.kernel_params_,2,X_proj,self.X_train_proj_,B=self.B_proj_)@self.C_proj_
            else:
                return self.kernel_func_(X_proj,self.X_train_proj_,**self.kernel_params_)@self.C_proj_

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
                'weights':self.weights,
                'matrix_kernel':self.matrix_kernel,
                'B':self.B}

    def set_params(self,**parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    