# config parameters for fitting dynamical systems model to manifest data
import numpy as np


drift_deg = 3
diff_deg = 0
param_deg_drift = 3
param_deg_diff = 3

# include sigmoid functions in the basis set or not
include_sigmoid = False
# right now, sigmoid functions have to be defined explicitly in the code
# otherwise, they are local to the function that defines them
# can't be pickled and saved to file
if include_sigmoid:
    n1 = 3
    def sigmoid_1(x):
        return 1/(1+np.exp(-n1*x))

    def sigmoid_1_string(x):
        return '1/(1+exp(-'+str(n1)+'*'+x+')'

    n2 = 4
    def sigmoid_2(x):
        return 1/(1+np.exp(-4*x))

    def sigmoid_2_string(x):
        return '1/(1+exp(-'+str(n2)+'*'+x+')'

    sigmoid_funcs = [sigmoid_1, sigmoid_2]
    func_names = [sigmoid_1_string, sigmoid_2_string]
else:
    sigmoid_funcs = []
    func_names = []