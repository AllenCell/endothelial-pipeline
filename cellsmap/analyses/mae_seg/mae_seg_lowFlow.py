# %%
import numpy as np

import matplotlib.pyplot as plt

import sympy

# in utils/langevin_sindy folder, includes all the langevin-regression code implemented for 2d
from cellsmap.analyses.utils.langevin_sindy.select_timelag import select_lag_2D
import cellsmap.analyses.utils.langevin_sindy.langevin_sindy as lg
import cellsmap.analyses.utils.langevin_sindy.timecorr as tc
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

# phase plane analysis code
import cellsmap.analyses.utils.pplane as pplane

# Euler-Maruyama for stochastic simulation of trajectories
from cellsmap.analyses.utils.stochastic_sim import stochastic_sim_EM

import cellsmap.analyses.utils.gen_potential as gp

from cellsmap.analyses.workflows.fit_SDE_model import get_scaled_traj
from cellsmap.analyses.utils.langevin_sindy.fit_langevin_sindy import get_bins, get_hist
from cellsmap.analyses.workflows.analyze_SDE_model import get_model_functions


# %%
path_to_data = '//allen/aics/assay-dev/users/Benji/cellsmap/results/mae_seg/predictions.csv'
metadata = ["crop_index","T"]
savedir = '/allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/mae_seg/'
PCA = True
ndim = 2
feats_to_analyze= None
log_file = None
X_t = get_scaled_traj(path_to_data,metadata,savedir,PCA,ndim,feats_to_analyze,log_file=log_file)

# %%
t_change = 283
dt = 5
X_t_low = X_t[:,t_change:,:]
num_traj = X_t_low.shape[0]
for i in range(num_traj):
    plt.plot(X_t_low[i,:,0],X_t_low[i,:,1],'k-',alpha=0.1)

# %%
# PDF of states
fig,ax = plt.subplots()
data = [X_t_low[i] for i in range(num_traj)]
data_stationary = [X_t_low[i,X_t_low.shape[1]//2:] for i in range(num_traj)]
N = (32,32)
bins, centers, dx = get_bins(ndim, data, N)
p_hist = get_hist(ndim, data_stationary, bins)
ax.imshow(p_hist.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap='inferno', aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]))

# %%

### Build SINDy libraries with sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')

nf=3
ns=1
f_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(nf+1) for k in range(m+1)]),2)  # Polynomial library for drift
s_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(ns+1) for k in range(m+1)]),2)  # Polynomial library for diffusion

Xi = np.load(savedir+'outputs/model_coeffs_low.npy')
V = np.load(savedir+'outputs/cost_vals_low.npy')

# %%
####################
# SSR cost function
####################

labels = [r'${0}$'.format(sympy.latex(t)) for t in np.concatenate((f_expr, s_expr))]

active = abs(Xi) > 1e-8

ndim=2
n_terms = len(labels)
plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.scatter(np.arange(len(V)), V, c='k')

plt.gca().set_xticks(np.arange(0,n_terms-(2*ndim-1),2))
plt.gca().set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
plt.xlabel('Sparsity')
plt.ylabel(r'Cost')
plt.gca().set_yscale('log')
plt.grid()

active_1 = np.concatenate((active[:len(f_expr)//2], active[len(f_expr):len(f_expr)+len(s_expr)//2]))
labels_1 = np.concatenate((labels[:len(f_expr)//2], labels[len(f_expr):len(f_expr)+len(s_expr)//2]))
plt.subplot(132)
plt.pcolor(active_1, cmap='bone_r', edgecolors='gray')
plt.gca().set_yticks(0.5+np.arange(active_1.shape[0]))
plt.gca().set_yticklabels(labels_1)
plt.gca().set_xticks(0.5+np.arange(0,n_terms-(2*ndim-1),2))
plt.gca().set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
plt.xlabel('Sparsity')
plt.ylabel('Active terms (f1, a1)')


active_2 = np.concatenate((active[len(f_expr)//2:len(f_expr)], active[len(f_expr)+len(s_expr)//2:]))
labels_2 = np.concatenate((labels[len(f_expr)//2:len(f_expr)], labels[len(f_expr)+len(s_expr)//2:]))
plt.subplot(133)
plt.pcolor(active_2, cmap='bone_r', edgecolors='gray')
plt.gca().set_yticks(0.5+np.arange(active_2.shape[0]))
plt.gca().set_yticklabels(labels_2)
plt.gca().set_xticks(0.5+np.arange(0,n_terms-(2*ndim-1),2))
plt.gca().set_xticklabels(np.arange(n_terms, (2*ndim-1), -2))
plt.xlabel('Sparsity')
plt.ylabel('Active terms (f2, a2)')

plt.show()

# %%
# Select model with the fewest terms before the cost function spikes
n_terms = 12
f, D, sigma = get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr)

# %%
X1,X2 = np.meshgrid(centers[0],centers[1])
f_vals = f(X1,X2)
D_vals = D(X1,X2)
# Compare PDFs: empirical vs Fokker-Planck solution with model
Nx = len(centers[0])
Ny = len(centers[1])
dx = [bins[0][1]-bins[0][0],bins[1][1]-bins[1][0]]
dt = 5
fp = fps.SteadyFP((Nx,Ny), dx)
p_fit = fp.solve(np.swapaxes(f_vals,1,2),np.swapaxes(D_vals,1,2))
print('KL divergence (LINDy model): {0:0.5f}'.format(tc.kl_divergence(p_hist, p_fit, dx=dx, tol=1e-6)))
# PDF of states
fig,ax = plt.subplots(1,2)
ax[0].imshow(p_hist.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap='inferno', aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]))

ax[1].imshow(p_fit.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap='inferno', aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]),vmin=0)
afp = fps.AdjFP(centers,ndim=2)
afp.precompute_operator(f_vals.transpose((0,2,1)).reshape((2,Nx*Ny)), D_vals.transpose((0,2,1)).reshape((2,Nx*Ny)))
stride=20
f_tau, D_tau = afp.solve(stride*dt,d=[0,1])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


# %%
# streamplot
plt.streamplot(X1,X2,f_vals[0],f_vals[1],color='k')


# %%
# phase portrait
f1 = lambda x1,x2: f(x1,x2)[0]
f2 = lambda x1,x2: f(x1,x2)[1]
xvec = np.linspace(-30,25,50)
yvec = np.linspace(-25,25,50)
fig = pplane.plot_portrait(f1,f2,xvec,yvec)#,ICs=[inits[:,i] for i in range(len(inits_idx))],tVec=np.linspace(0,dt*t_change,100))

# %%
#### Potential U, vector field decomposition

xvec = np.linspace(-30,25,100)
yvec = np.linspace(-25,25,100)
centers_new = [xvec,yvec]

X1,X2 = np.meshgrid(xvec,yvec)
f_vals_new = f(X1,X2)
D_vals_new = D(X1,X2)

# %%
# Potential U, vector field decomposition
U, grad_term, flux_term = gp.grad_flux_decomposition(np.swapaxes(f_vals_new,1,2),np.swapaxes(D_vals_new,1,2),centers_new)

# %%
grad = grad_term.copy()
# normalize gradient
#grad = grad/np.sqrt(grad[0]**2+grad[1]**2)

flux = flux_term.copy()
# normalize flux
#flux = flux/np.sqrt(flux[0]**2+flux[1]**2)

# %%
fig, ax = plt.subplots()

# plot potential and gradient/flux fields
im = ax.imshow(U.T,interpolation='nearest', origin='lower',
           extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
           cmap='jet', aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))

downsample=8
ydownsample=12
plt.quiver(xvec[::downsample],yvec[::ydownsample],grad[0][::downsample,::ydownsample].T,grad[1][::downsample,::ydownsample].T,color='w',pivot='tail')
plt.quiver(xvec[::downsample],yvec[::ydownsample],flux[0][::downsample,::ydownsample].T,flux[1][::downsample,::ydownsample].T,color='m',pivot='tail')
plt.xlabel('PC1')
plt.ylabel('PC2')
fig.colorbar(im,label='$-\ln P$')

# %%

x_, y_ = np.meshgrid(xvec,yvec,indexing='ij')

fig = plt.figure(figsize=plt.figaspect(1/3))
ax1 = fig.add_subplot(1,2,1, projection='3d')
surf = ax1.plot_surface(x_,y_, U, cmap='jet')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('$-\ln P$')
plt.tight_layout()

# %%
# plot magnitude of flux term
flux_magnitude = np.sqrt(flux_term[0]**2+flux_term[1]**2)
fig, ax = plt.subplots()
im = ax.imshow(flux_magnitude.T,interpolation='nearest', origin='lower',
           extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
           cmap='jet', aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))
fig.colorbar(im,label='$||\\text{flux}||$')

# %%
