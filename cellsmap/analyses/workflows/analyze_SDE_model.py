import numpy as np
import sympy

from cellsmap.util import io

def lambdify_SINDy(coeff,f_expr,vars):
    '''Lambdify a sympy function from SINDy coefficients (coeff) and function library (f) defined on given sympy variables (vars).'''
    return sympy.lambdify(vars, coeff.dot(f_expr), ["scipy", "numpy"])

def vec_field_2D(f1,f2,X1,X2):
    '''Create a 2D vector field f=(f1,f2) from component functions f1 and f2.'''
    F1 = f1(X1,X2)
    F2 = f2(X1,X2)
    if np.isscalar(F1):
        F1 = F1 + 0*X1
    if np.isscalar(F2):
        F2 = F2 + 0*X1
    return np.array([F1,F2])

def get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr):
    Xi_f = Xi[:len(f_expr), (2*ndim-1)-n_terms]
    Xi_s = Xi[len(f_expr):, (2*ndim-1)-n_terms]
    if ndim == 1:
        print("SINDy expression (drift): ")
        print("     f(x) = ", np.round(Xi_f,4).dot(f_expr))
        print("SINDy expression (diffusion): ")
        print("     sigma(x) = ", np.round(Xi_s,4).dot(s_expr))
        # lambdify SINDy expressions
        x = sympy.symbols('x')
        #drift
        f = lambdify_SINDy(Xi_f, f_expr, x)
        # diffusion
        sigma = lambdify_SINDy(Xi_s, s_expr, x)
        D = lambda x: 0.5*(sigma(x))**2
    else: # 2D
        print("SINDy expression (drift): ")
        print("     f_1(x1,x2) = ", np.round(Xi_f[:len(f_expr)//2],4).dot(f_expr[:len(f_expr)//2]))
        print("     f_2(x1,x2) = ", np.round(Xi_f[len(f_expr)//2:],4).dot(f_expr[len(f_expr)//2:]))
        print("SINDy expression (diffusion): ")
        print("     sigma_1(x1,x2) = ", np.round(Xi_s[:len(s_expr)//2],4).dot(s_expr[:len(s_expr)//2]))
        print("     sigma_2(x1,x2) = ", np.round(Xi_s[len(s_expr)//2:],4).dot(s_expr[len(s_expr)//2:]))
        # lambdify SINDy expressions
        x1 = sympy.symbols('x1')
        x2 = sympy.symbols('x2')
        #drift
        f1 = lambdify_SINDy(Xi_f[:len(f_expr)//2], f_expr[:len(f_expr)//2], [x1,x2])
        f2 = lambdify_SINDy(Xi_f[len(f_expr)//2:], f_expr[len(f_expr)//2:], [x1,x2])

        # diffusion
        sigma1 = lambdify_SINDy(Xi_s[:len(s_expr)//2], s_expr[:len(s_expr)//2], [x1,x2])
        sigma2 = lambdify_SINDy(Xi_s[len(s_expr)//2:], s_expr[len(s_expr)//2:], [x1,x2])

        D1 = lambda x1,x2: 0.5*(sigma1(x1,x2))**2
        D2 = lambda x1,x2: 0.5*(sigma2(x1,x2))**2

        # turn into vector field that can be evaluated on a grid
        f = lambda X1,X2: vec_field_2D(f1,f2,X1,X2)
        D = lambda X1,X2: vec_field_2D(D1,D2,X1,X2)
        sigma = lambda X1,X2: vec_field_2D(sigma1,sigma2,X1,X2)
    return f, D, sigma

def load_model_outputs(savedir,ndim,flow='all'):
    '''Load SINDy model outputs from saved directory.'''
    coeff_file = savedir+'outputs/model_coeffs_'+flow+'.npy'
    cost_file = savedir+'outputs/cost_vals_'+flow+'.npy'
    Xi = np.load(coeff_file)
    V = np.load(cost_file)
    f_expr = np.load(savedir+'outputs/f_expr.npy')
    s_expr = np.load(savedir+'outputs/s_expr.npy')
    return V, Xi, f_expr, s_expr

# define function to compare histogram to stationary FP solution

# define function to plot phase portrait of vector field f

# define function to plot generalized potential energy landscape

def main(n_terms, ndim, savedir, flow='all'):
    '''Main function to load and analyze SINDy model outputs.'''
    # load SINDy model outputs
    V, Xi, f_expr, s_expr = load_model_outputs(savedir,ndim,flow)
    print("Cost at optimal sparsity: ", V[(2*ndim-1)-n_terms])
    f, D, sigma = get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr)

    # load histogram data to compare to stationary Fokker Planck solution

    # plot phase portrait of vector field f OR phase line plot of drift function f (write function to do phase line part)

    # plot generalized potential energy landscape (write function to do this in 1D as well)

    return f, D, sigma

if __name__ == 'main':
    config_name = input("Enter the name of the configuration in `dynamics_config.yaml` used to fit the model: ")
    config_inputs = io.get_dynamics_inputs(config_name)
    savedir = config_inputs[-1]
    ndim = config_inputs[2]
    split_high_low = config_inputs[6]

    if split_high_low:
        flow = input("Enter the flow regime to analyze (high or low): ")

    n_terms = int(input("Select optimal sparsity for Langevin Regression model (see plot of cost function): "))

    main(n_terms, ndim, savedir, flow)
    




        