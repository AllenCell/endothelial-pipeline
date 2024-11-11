# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from IPython.display import HTML
from matplotlib import animation
from cellsmap.analyses.workflows.analyze_SDE_model import load_model_outputs, get_model_functions, get_histogram_data, compare_hist_to_FP, plot_phase, plot_gen_potential
import cellsmap.analyses.utils.stochastic_sim as em_sim
import cellsmap.analyses.utils.gen_potential as gp
# %%

savedir = "//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/20240305/cdh5_small_patch/"
flow = 'low'
ndim=2
n_terms = 14
V, Xi, f_expr, s_expr = load_model_outputs(savedir,flow)
print("Cost at optimal sparsity: ", V[(2*ndim-1)-n_terms])
f, D, sigma = get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr)

# load histogram data to compare to stationary Fokker Planck solution
p_hist, bins, centers = get_histogram_data(ndim,savedir,flow)
centers_new = centers.copy()

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
plot_phase(ndim,f,centers_new,savedir,flow)

# %%

N_fine = 125
if ndim == 1:
    x_fine = np.linspace(centers_new[0],centers_new[-1],N_fine)
    centers_new = [x_fine]
    f_vals_new = f(x_fine)
    D_vals_new = D(x_fine)
else: # 2D
    x_fine = np.linspace(centers_new[0][0],centers_new[0][-1],N_fine)
    y_fine = np.linspace(centers_new[1][0],centers_new[1][-1],N_fine)
    centers_fine = [x_fine,y_fine]
    X1,X2 = np.meshgrid(x_fine,y_fine)
    f_vals_new = np.swapaxes(f(X1,X2),1,2)
    D_vals_new = np.swapaxes(D(X1,X2),1,2)
print('**** Plotting generalized potential energy landscape **** \n')
U, grad_term, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine)
grad_norm = grad_term/(np.sqrt(grad_term[0]**2+grad_term[1]**2))
flux_norm = flux_term/(np.sqrt(flux_term[0]**2+flux_term[1]**2))
fig, ax = plot_gen_potential(ndim,U,centers_fine,savedir,surf=True,flow=flow)

# %%
p_hist_high, bins_high, centers_high = get_histogram_data(ndim,savedir,'high')
cdf = np.cumsum(p_hist_high.ravel())
cdf = cdf / cdf[-1]
values = np.random.rand(1)
value_bins = np.searchsorted(cdf, values)
x_idx, y_idx = np.unravel_index(value_bins,
                                (len(centers_high[0]), len(centers_high[1])))
random_from_cdf = np.column_stack((centers_high[0][x_idx], centers_high[1][y_idx]))[0][:,None]
print(random_from_cdf)

# %%
em_traj = em_sim.stochastic_sim_EM(random_from_cdf, lambda x: f(x[0],x[1]), lambda x: sigma(x[0],x[1]), 300, 5)
# %%
plt.plot(em_traj[0,:,0],em_traj[1,:,0])
# %%

line, = ax.plot([], [], lw=2, color='white')

def init():
    line.set_data([], [])
    return line,

def animate(i, line, X, Y):
    line.set_data(X[:i], Y[:i])
    return line,

# %%
anim = animation.FuncAnimation(fig, animate, init_func=init, fargs=(line, em_traj[0,:,0], em_traj[1,:,0]),
                           frames=10, interval=200,
                           repeat_delay=5, blit=True)
plt.show()
# %%
HTML(anim.to_jshtml())
# %%
