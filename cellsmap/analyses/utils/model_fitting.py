import numpy as np
import pysindy as ps
from typing import Callable

def make_sigmoid(n:int) -> Callable:
    # callable function for adding sigmoid functions to SINDy library
    def _(x):
        return 1/(1+np.exp(-n*x))
    return _

def make_sigmoid_string(n:int) -> Callable:
    # string representation of sigmoid function
    def _(x):
        return '1/(1+exp(-'+str(n)+'*'+x+')'
    return _

def build_drift_lib(ndim:int,drift_deg:int=3,param_deg:int=3) -> ps.ParameterizedLibrary:
    # build set of basis functions for regression model for drift term of SDE model
    drift_feature_lib = ps.PolynomialLibrary(degree=drift_deg, include_bias=True)

    # library for model dependence on control parameters (shear stress)
    drift_parameter_lib=ps.PolynomialLibrary(degree=param_deg, include_bias=True) # library for model dependence on control parameters (shear stress)

    # build full library for drift term: pySINDy parameterized library
    drift_lib=ps.ParameterizedLibrary(feature_library=drift_feature_lib,
        parameter_library=drift_parameter_lib,num_features=ndim,num_parameters=1) 
    
    return drift_lib

def build_diff_lib(ndim:int,diff_deg:int=0,param_deg=3) -> ps.ParameterizedLibrary:
    # build set of basis functions for regression model for diffusion term of SDE model

    diff_feature_lib=ps.PolynomialLibrary(degree=diff_deg, include_bias=True)
    diff_parameter_lib=ps.PolynomialLibrary(degree=param_deg, include_bias=False)
    diff_lib=ps.ParameterizedLibrary(feature_library=diff_feature_lib,
        parameter_library=diff_parameter_lib,num_features=ndim,num_parameters=1)
    
    return diff_lib