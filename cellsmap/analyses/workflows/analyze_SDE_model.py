import numpy as np
import sympy

from matplotlib import cm, colors
import matplotlib.pyplot as plt

from cellsmap.util import io
import cellsmap.analyses.utils.langevin_sindy.timecorr as tc
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps
import cellsmap.analyses.utils.viz as viz
import cellsmap.analyses.utils.pplane as pplane
import cellsmap.analyses.utils.gen_potential as gp

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

def load_model_outputs(savedir,flow='all'):
    '''Load SINDy model outputs from saved directory.'''
    coeff_file = savedir+'outputs/model_coeffs_'+flow+'.npy'
    cost_file = savedir+'outputs/cost_vals_'+flow+'.npy'
    Xi = np.load(coeff_file)
    V = np.load(cost_file)
    f_expr = np.load(savedir+'outputs/f_expr.npy',allow_pickle=True)
    s_expr = np.load(savedir+'outputs/s_expr.npy',allow_pickle=True)
    return V, Xi, f_expr, s_expr

def get_histogram_data(ndim,savedir,flow='all'):
    '''Load histogram data to compare to stationary Fokker Planck solution.'''
    p_hist = np.load(savedir+'outputs/histogram_'+flow+'.npy')
    if ndim == 1:
        bins = np.load(savedir+'outputs/histogram_bins_'+flow+'.npy')
        centers = 0.5*(bins[1:]+bins[:-1])
    else: # 2D
        bins_ = np.load(savedir+'outputs/histogram_bins_'+flow+'.npy',allow_pickle=True)
        bins = [bins_[i].astype(float) for i in range(len(bins_))] # convert from object to float
        centers=[0.5*(bins[0][1:]+bins[0][:-1]),0.5*(bins[1][1:]+bins[1][:-1])]
    return p_hist, bins, centers

def compare_hist_to_FP(ndim, p_hist, bins, f, D, savedir, flow='all'):
    '''Compare histogram to stationary Fokker Planck solution.'''
    # need to write 1D version of this function
    if ndim == 1:
        N = len(bins)-1 # number of bins
        dx = bins[1]-bins[0]
    else: # 2D
        N = (len(bins[0])-1, len(bins[1])-1) # number of bins in each dimension
        dx = [bins[0][1]-bins[0][0],bins[1][1]-bins[1][0]]
    fp = fps.SteadyFP(N, dx) # initialize stationary Fokker-Planck solver
    p_fit = fp.solve(f,D) # solve stationary Fokker-Planck equation
    p_fit[p_fit<1e-10] = 1e-10 # set small values to a small number to avoid numerical issues

    if ndim == 1:
        fig,ax= viz.init_plot()
        ax = viz.plot_histogram_1D(ax,p_hist,bins,color='k') # plot empirical PDF
        ax = viz.plot_histogram_1D(ax,p_fit,bins,color='#5891BF') # plot model PDF
    else:
        fig,ax = viz.init_subplots(1,2,figsize=(12,4))
        ax[0] = viz.plot_histogram_2D(ax[0],p_hist,bins,cmap='inferno') # plot empirical PDF
        ax[1] = viz.plot_histogram_2D(ax[1],p_fit,bins,cmap='inferno') # plot model PDF

    viz.save_plot(fig,savedir+'figs/histogram_comparison_'+flow) # save plot

    return tc.kl_divergence(p_hist, p_fit, dx=dx, tol=1e-7) # compare empirical and model PDFs via KL divergence

def plot_phase(ndim,f,centers,savedir,flow='all'):
    if ndim == 1:
        xvec = np.linspace(centers[0][0],centers[0][-1],50)
        fig, ax = pplane.phase_line(f,xvec)
    else: # 2D
        f1 = lambda x1,x2: f(x1,x2)[0]
        f2 = lambda x1,x2: f(x1,x2)[1]
        xvec = np.linspace(centers[0][0],centers[0][-1],50) # change this to take input? maybe a config file?
        yvec = np.linspace(centers[1][0],centers[1][-1],50)
        fig, ax = pplane.phase_portrait(f1,f2,xvec,yvec)
    viz.save_plot(fig,savedir+'figs/phase_portrait_'+flow)
    return

# define function to plot generalized potential energy landscape
def plot_gen_potential(ndim,U,centers,savedir,surf=False,flow='all'):
    if ndim==1:
        fig,ax = viz.plot_gen_potential_1D(U,centers)
        viz.save_plot(fig,savedir+'figs/gen_potential_'+flow)
    else: # 2D
        fig,ax = viz.plot_gen_potential_2D(U,centers[0],centers[1],cmap='jet',surf=surf)
        if surf:
            viz.save_plot(fig,savedir+'figs/gen_potential_surf_'+flow)
        else:
            viz.save_plot(fig,savedir+'figs/gen_potential_'+flow)
    return fig,ax

def main(n_terms, ndim, savedir, flow='all'):
    '''Main function to load and analyze SINDy model outputs.'''
    # load SINDy model outputs
    V, Xi, f_expr, s_expr = load_model_outputs(savedir,flow)
    print("Cost at optimal sparsity: ", V[(2*ndim-1)-n_terms])
    f, D, sigma = get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr)

    # load histogram data to compare to stationary Fokker Planck solution
    p_hist, bins, centers = get_histogram_data(ndim,savedir,flow)

    # following needs to be generalized to 1D
    if ndim == 1:
        f_vals = f(centers)
        D_vals = D(centers)
    else: #2D
        X1,X2 = np.meshgrid(centers[0],centers[1])
        f_vals = np.swapaxes(f(X1,X2),1,2)
        D_vals = np.swapaxes(D(X1,X2),1,2)
    # Compare PDFs: empirical vs Fokker-Planck solution with model
    print('**** Comparing empirical PDF to stationary Fokker-Planck solution with Langevin Regression model ****')
    kl_div = compare_hist_to_FP(ndim, p_hist, bins, f_vals, D_vals,savedir,flow)
    print('KL divergence between histogram of "stationary" data and stationary pdf of Langevin Regression model: {0:0.5f}'.format(kl_div)) # compare empirical and model PDFs

    # plot phase portrait of vector field f OR phase line plot of drift function f
    print("\n","**** Plotting phase portrait of drift function f **** \n",sep="")
    plot_phase(ndim,f,centers,savedir,flow)

    N_fine = 125
    if ndim == 1:
        x_fine = np.linspace(centers[0],centers[-1],N_fine)
        centers_new = [x_fine]
        f_vals_new = f(x_fine)
        D_vals_new = D(x_fine)
    else: # 2D
        x_fine = np.linspace(centers[0][0],centers[0][-1],N_fine)
        y_fine = np.linspace(centers[1][0],centers[1][-1],N_fine)
        centers_new = [x_fine,y_fine]
        X1,X2 = np.meshgrid(x_fine,y_fine)
        f_vals_new = np.swapaxes(f(X1,X2),1,2)
        D_vals_new = np.swapaxes(D(X1,X2),1,2)
    print('**** Plotting generalized potential energy landscape **** \n')
    U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_new)
    fig, ax = plot_gen_potential(ndim,U,centers_new,savedir,surf=True,flow=flow)
    # same but with vector field decomposition
    fig, ax = plot_gen_potential(ndim,U,centers_new,savedir,flow=flow)
    downsample=8
    ax.quiver(x_fine[::downsample],y_fine[::downsample],grad_term[0][::downsample,::downsample].T,grad_term[1][::downsample,::downsample].T,scale=2,color='w',pivot='tail')
    ax.quiver(x_fine[::downsample],y_fine[::downsample],flux_term[0][::downsample,::downsample].T,flux_term[1][::downsample,::downsample].T,scale=20,color='m',pivot='tail')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    viz.save_plot(fig,savedir+'figs/gen_potential_decomp_'+flow+'.png')
    # in 2D, can also add vector field decomposition to this plot -- write up later

    return

if __name__ == '__main__':
    print(r"""
   ______________    __   _____ __  ______    ____ 
  / ____/ ____/ /   / /  / ___//  |/  /   |  / __ \
 / /   / __/ / /   / /   \__ \/ /|_/ / /| | / /_/ /
/ /___/ /___/ /___/ /______/ / /  / / ___ |/ ____/ 
\____/_____/_____/_____/____/_/  /_/_/  |_/_/      
                                                                                               
          """)
    print("******* Cellsmap dynamical systems model analysis workflow ******* \n")
    config_name = input("Enter the name of the configuration in `dynamics_config.yaml` used to fit the model: ")
    config_inputs = io.get_dynamics_inputs(config_name)
    savedir = config_inputs[-2]
    ndim = config_inputs[2]
    split_high_low = config_inputs[6]

    if split_high_low:
        flow = input("Enter the flow regime to analyze (high or low): ")

    n_terms = int(input("Select optimal sparsity for Langevin Regression model (see plot of cost function): "))

    main(n_terms, ndim, savedir, flow)
    




        