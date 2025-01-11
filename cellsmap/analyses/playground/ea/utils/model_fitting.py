import numpy as np

import cellsmap.analyses.utils.kernel_regression as kr
import pysindy as ps

# initialize with what datasets to train on?
class LangevinDynamicsModel:
    def __init__(self,method='kernel_regression'):
        self.method = method
        if method == 'kernel_regression':
            self.regression_model = kr.KernelRegression(beta=0.01)
        elif method == 'sindy':
            self.regression_model = ps.SINDy(feature_library = ps.PolynomialLibrary(degree=3), optimizer = ps.SSR())
        else:
            raise ValueError('Specified nonlinear regression method not implemented.')
        
    def model_fit_inputs(self, X, Y, u=None):
        if self.method == 'kernel_regression':
            return {'X':X, 'Y':Y, 'u':u}
        elif self.method == 'sindy':
            return {'x':X, 'x_dot':Y, 'u':u}
        
    def fit_drift(self, X, f_KM, u=None):
        model_inputs = self.model_fit_inputs(X, f_KM, u)
        drift_model = self.regression_model.fit(*model_inputs)
        return
    
    def fit_diffusion(self, X, D_KM, u=None):
        model_inputs = self.model_fit_inputs(X, D_KM, u)
        diffusion_model = self.regression_model.fit(*model_inputs)
        return
    
    def fit(self, X, f_KM, D_KM, u=None):
        self.fit_drift(X, f_KM, u)
        self.fit_diffusion(X, D_KM, u)
        self.langevin_model = {'drift':self.drift_model, 'diffusion':self.diffusion_model}
        return
    
