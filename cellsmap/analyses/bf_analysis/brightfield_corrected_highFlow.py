# %%
import numpy as np

import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 2**128

import matplotlib.pyplot as plt
import sys

# from fplanck import fokker_planck, boundary

from sklearn.neighbors import KernelDensity

from scipy import constants

import sympy

# in utils/langevin_sindy folder, includes all the langevin-regression code implemented for 2d
import cellsmap.analyses.utils.langevin_sindy.langevin_sindy as lg
import cellsmap.analyses.utils.langevin_sindy.timecorr as tc
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps

# phase plane analysis code
import cellsmap.analyses.utils.pplane as pplane

# Euler-Maruyama for stochastic simulation of trajectories
from cellsmap.analyses.utils.stochastic_sim import stochastic_sim_EM

import cellsmap.analyses.utils.gen_potential as gp

# %%
## Load bright field MAE data

# Masked autoencoder dataset info:
# - columns 0-255 are the latent features of the contrastive model
# - location gives a unique id for each 512x512 patch of the image (stitched FOVs) that is consistent across time (54 locations)
# - time is the index into the movie (0 to 576, units 5 minutes)

# Original endothelial cell dataset info (from `cellsmap/cellsmap/data_config.yaml`):
# - flow rate of 20 dyn/cm^2 from 0 to 24 hours
# - flow rate of 6 dyn/cm^2 from 24 to 48 hours
# - time interval between images (data points) is 5 minutes

# Data loaded and preprocessed in `mae_feats_init.py`.
exp_var = np.load('../data/bf_ExpVar.npy')
pcs = np.load('../data/bf_PCs.npy')

num_modes_95 = np.where(np.cumsum(exp_var) > 0.95)[0].min()
print("Number of modes to explain 95% of variance: ", num_modes_95)
X_t = np.load('../data/bf_95pctVarPCs_all.npy') # preprocessed data: num_loc * num_timepoints * num_modes_95 array

X_t_high = np.load('../data/bf_95pctVarPCs_highFlow.npy')
X_t_low = np.load('../data/bf_95pctVarPCs_lowFlow.npy')

num_loc = X_t.shape[0]
num_t = X_t.shape[1]
t_change = (24*60 - 25)//5 # time point (frame number) at which to change from high to low flow occurs (25 minutes before 24 hours)
dt=5

# %%
fig,ax = plt.subplots()
# plot top PCA mode vs time for each location at high flow
for i in range(num_loc):
    ax.plot(5*np.arange(num_t)/60,X_t[i,:,0]-X_t[i,0,0],'k-',alpha=0.25,linewidth=1)
ax.set_xlim([0,(num_t)*5//60])
#plt.vlines(5*t_change/60,-20,25,color='r',linestyles='dashed')
#plt.vlines(530/60,-20,25,color='b',linestyles='dashed')
ax.set_xlabel("time (hours)", fontsize=16)
ax.set_ylabel("PC1", fontsize=16)
# %%
# plot 2nd PCA mode vs time for each location at high flow
fig,ax = plt.subplots()
for i in range(num_loc):
    ax.plot(5*np.arange(num_t)/60,X_t[i,:,1]-X_t[i,0,1],'k-',alpha=0.25,linewidth=1)
ax.set_xlim([0,(num_t)*5//60])
#plt.vlines(5*t_change/60,-30,20,color='r',linestyles='dashed')
ax.set_xlabel("time (hours)", fontsize=16)
ax.set_ylabel("PC2", fontsize=16)

# %%
# plot top two PC modes for each location at high flow
fig,ax = plt.subplots()
for i in range(num_loc):
    ax.plot(X_t[i,:,0]-X_t[i,0,0],X_t[i,:,1]-X_t[i,0,1],'k-',alpha=0.25,linewidth=1)
ax.set_xlabel("PC1", fontsize=16)
ax.set_ylabel("PC2", fontsize=16)

# %%
# Corrected features: Langevin regression (2D)
### High flow trajectories
X_t_high = np.load('../data/bf_95pctVarPCs_highFlow.npy')
num_loc = X_t_high.shape[0]
dt = 5

# correct for bias in x position of patch, subtract off X(0)
for i in range(num_loc):
    X_t_high[i,:,:] = X_t_high[i,:,:] - X_t_high[i,0,:]
# Plot truth, fast sampling, slow sampling

# %%
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

data = [X_t_high[i,:,:2] for i in range(num_loc)] # high flow data, pass as list into KM_avg
data_stationary = [X_t_high[i,100:,:] for i in range(num_loc)] # "Steady state" low flow data, for histogram
data_stationary_high = data_stationary

Nx = 32
min0 = min([min(traj[:,0]) for traj in data])
max0 = max([max(traj[:,0]) for traj in data])
bin0_min = 0.5*(np.floor(min0)+np.round(min0,1))
bin0_max = 0.5*(np.ceil(max0)+np.round(max0,1))
bins0 = np.linspace(bin0_min, bin0_max, Nx+1)
centers0 = 0.5*(bins0[1:]+bins0[:-1])

Ny=24
min1 = min([min(traj[:,1]) for traj in data])
max1 = max([max(traj[:,1]) for traj in data])
bin1_min = 0.5*(np.floor(min1)+np.round(min1,1))
bin1_max = 0.5*(np.ceil(max1)+np.round(max1,1))
bins1 = np.linspace(bin1_min, bin1_max, Ny+1)
centers1 = 0.5*(bins1[1:]+bins1[:-1])

dx = [bins0[1]-bins0[0],bins1[1]-bins1[0]]

bins = [bins0,bins1]
centers = [centers0,centers1]

dt=5
stride=9


f_fine, a_fine, _,_ = lg.KM_avg_2D(data, bins, stride=1, dt=dt, multi_traj=True)

f_coarse, a_coarse, _,_ = lg.KM_avg_2D(data, bins, stride=stride, dt=dt,multi_traj=True)
x1,x2 = np.meshgrid(centers0,centers1)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.quiver(x1,x2, f_fine[:,:,0].T,f_fine[:,:,1].T,color=colors[0], label=r'$\tau=$'+str(np.round(dt,2)))
plt.quiver(x1,x2, f_coarse[:,:,0].T,f_coarse[:,:,1].T,color=colors[1], label=r'$\tau=$'+str(np.round(dt*stride,2)))
plt.legend(fontsize=14)
plt.title('Drift')
plt.xlabel('$x_1$', fontsize = 24)
plt.ylabel('$x_2$', fontsize = 24)
plt.grid()

plt.subplot(122)
plt.quiver(x1,x2, a_fine[:,:,0].T,a_fine[:,:,1].T,color=colors[0])
plt.quiver(x1,x2, a_coarse[:,:,0].T, a_coarse[:,:,1].T, color=colors[1])
plt.title('Diffusion')
plt.xlabel('$x_1$', fontsize = 24)
plt.ylabel('$x_2$', fontsize = 24)
plt.grid()

plt.subplots_adjust(wspace=0.3)
plt.show()
# %%
# PDF of states
fig,ax = plt.subplots()
p_hist, _, _ = np.histogram2d(np.concatenate(data_stationary)[:,0],np.concatenate(data_stationary)[:,1], bins, density=True)
ax.imshow(p_hist.T,interpolation='nearest', origin='lower',
           extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]],
           cmap='inferno', aspect=(bins[0][-1]-bins[0][0])/(bins[1][-1]-bins[1][0]))


# %%
# autocorrelation function (average across all locations)
tau = dt*np.arange(0, data[0].shape[0])
acf = np.zeros((len(tau),2,2))
for loc_idx in range(num_loc):
    acf = acf + tc.autocorr_func_2D(data[loc_idx])
acf = acf/num_loc

fig, axs = plt.subplots(1,3, figsize=(18, 4))
tup_list = [(0,0),(0,1),(1,1)]
for ii in range(3):
    i,j = tup_list[ii]
    axs[ii].plot(tau, acf[:,i,j], 'k')
    axs[ii].set_ylabel(r'Autocorrelation $C_{('+str(i+1)+','+str(j+1)+')}(\\tau)$')
    axs[ii].set_xlabel(r'Time lag $\tau$')
    axs[ii].vlines(stride*dt, acf.min()-0.1, acf.max()+0.1, 'r', '--')
    axs[ii].set_ylim([-0.05, 1.15])
    axs[ii].set_xlim([0.5*dt, 1e3])
    axs[ii].set_xscale('log')
    axs[ii].grid()

# %%
# Markov test
lag = np.round( np.logspace(0.1, 2, 100) ).astype(int)
kl_div = np.zeros((num_loc,len(lag)))
for loc_idx in range(num_loc):
    kl_div[loc_idx,:] = np.array([tc.markov_test(data[loc_idx][:,0], delta, N=Nx) for delta in lag])
kl_div = np.nanmean(kl_div,axis=0)

plt.figure(figsize=(8, 3))
ax = plt.gca()
ax.set_xscale('log')
ax.plot(dt*lag, kl_div, 'k.')
ax.vlines(dt*stride, -0.3, 1.2, 'r', '--')

ax.set_ylabel(r'$\mathcal{D}_{KL}(\tau)$')
ax.set_xlabel(r'Sampling time $\tau$')
ax.set_xlim([dt*lag.min()-0.05, dt*lag.max()+0.05])
ax.set_ylim([1e-2, np.max([np.nanmax(kl_div)+0.05,1])])
plt.grid()

# %%

# Load outputs from `langevin_2D_bfcorr_highFlow.py`

## load Kramers-Moyal averages
f_KM = np.load('../outputs/KM_drift_bfcorr_highFlow.npy')

a_KM = np.load('../outputs/KM_diff_bfcorr_highFlow.npy')

f_err = np.load('../outputs/KM_drift_err_bfcorr_highFlow.npy')

a_err = np.load('../outputs/KM_diff_err_bfcorr_highFlow.npy')
### Build SINDy libraries with sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')

nf=3
ns=1
f_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(nf+1) for k in range(m+1)]),2)  # Polynomial library for drift
s_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(ns+1) for k in range(m+1)]),2)  # Polynomial library for diffusion

Xi_high = np.load('../outputs/coeffs_bfcorr_highFlow.npy')
V_high = np.load('../outputs/cost_bfcorr_highFlow.npy')

# %%
####################
# SSR cost function
####################

labels = [r'${0}$'.format(sympy.latex(t)) for t in np.concatenate((f_expr, s_expr))]

active = abs(Xi_high) > 1e-8

ndim=2
n_terms = len(labels)
plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.scatter(np.arange(len(V_high)), V_high, c='k')

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
n_terms = 16
print("Optimal sparsity: ", n_terms)
print("Cost at optimal sparsity: ", V_high[(2*ndim-1)-n_terms])
Xi_f_high = Xi_high[:len(f_expr), (2*ndim-1)-n_terms]
Xi_s_high = Xi_high[len(f_expr):, (2*ndim-1)-n_terms]
print("SINDy expression (drift): ")
print("     f_1(x1,x2) = ", np.round(Xi_f_high[:len(f_expr)//2],4).dot(f_expr[:len(f_expr)//2]))
print("     f_2(x1,x2) = ", np.round(Xi_f_high[len(f_expr)//2:],4).dot(f_expr[len(f_expr)//2:]))
print("SINDy expression (diffusion): ")
print("     sigma_1(x1,x2) = ", np.round(Xi_s_high[:len(s_expr)//2],4).dot(s_expr[:len(s_expr)//2]))
print("     sigma_2(x1,x2) = ", np.round(Xi_s_high[len(s_expr)//2:],4).dot(s_expr[len(s_expr)//2:]))
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')
f1_high = sympy.lambdify([x1,x2], Xi_f_high[:len(f_expr)//2].dot(f_expr[:len(f_expr)//2]))
f2_high = sympy.lambdify([x1,x2], Xi_f_high[len(f_expr)//2:].dot(f_expr[len(f_expr)//2:]))

a1_high = sympy.lambdify([x1,x2], 0.5*(Xi_s_high[:len(s_expr)//2].dot(s_expr[:len(s_expr)//2]))**2)
a2_high = sympy.lambdify([x1,x2], 0.5*(Xi_s_high[len(s_expr)//2:].dot(s_expr[len(s_expr)//2:]))**2)

sigma1_high = sympy.lambdify([x1,x2], Xi_s_high[:len(s_expr)//2].dot(s_expr[:len(s_expr)//2]))
sigma2_high = sympy.lambdify([x1,x2], Xi_s_high[len(s_expr)//2:].dot(s_expr[len(s_expr)//2:]))

def f_high(X1,X2):
    F1 = f1_high(X1,X2)
    #F1 = np.array(F1.tolist(),dtype=float)
    F2 = f2_high(X1,X2)
    #F2 = np.array(F2.tolist(),dtype=float)
    if np.isscalar(F1):
        F1 = F1 + 0*X1
    if np.isscalar(F2):
        F2 = F2 + 0*X1
    return np.array([F1,F2])

def a_high(X1,X2):
    A1 = a1_high(X1,X2)
    #A1 = np.array(A1.tolist(),dtype=float)
    A2 = a2_high(X1,X2)
    #A2 = np.array(A2.tolist(),dtype=float)
    if np.isscalar(A1):
        A1 = A1 + 0*X1
    if np.isscalar(A2):
        A2 = A2 + 0*X1
    return np.array([A1,A2])

def sigma_high(X1,X2):
    S1 = sigma1_high(X1,X2)
    #S1 = np.array(S1.tolist(),dtype=float)
    S2 = sigma2_high(X1,X2)
    #S2 = np.array(S2.tolist(),dtype=float)
    if np.isscalar(S1):
        S1 = S1 + 0*X1
    if np.isscalar(S2):
        S2 = S2 + 0*X1
    return np.array([S1,S2])

# %%
X1,X2 = np.meshgrid(centers[0],centers[1])
f_vals = f_high(X1,X2)
a_vals = a_high(X1,X2)
# Compare PDFs: empirical vs Fokker-Planck solution with model
fp = fps.SteadyFP((Nx,Ny), dx)
p_fit = fp.solve(np.swapaxes(f_vals,1,2),np.swapaxes(a_vals,1,2))
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
afp.precompute_operator(f_vals.transpose((0,2,1)).reshape((2,Nx*Ny)), a_vals.transpose((0,2,1)).reshape((2,Nx*Ny)))
f_tau, a_tau = afp.solve(stride*dt,d=[0,1])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

# %%

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.quiver(X1,X2, f_KM[:,:,0].T,f_KM[:,:,1].T,color=colors[0], label=r'K-M ($\tau=$'+str(np.round(stride*dt,2))+')')
plt.quiver(X1,X2, f_tau[0].reshape((Nx,Ny)).T,f_tau[1].reshape((Nx,Ny)).T,color=colors[1], label=r'Model ($\tau=$'+str(np.round(stride*dt,2))+')')
plt.quiver(X1,X2, f_vals[0],f_vals[1],color=colors[2], label=r'Model ($\tau=0$)')
plt.title('Drift')
plt.xlabel('$x_1$', fontsize = 24)
plt.ylabel('$x_2$', fontsize = 24)
plt.grid()

plt.subplot(122)
plt.quiver(X1,X2, a_KM[:,:,0].T,a_KM[:,:,1].T,color=colors[0], label=r'K-M ($\tau=$'+str(np.round(stride*dt,2))+')')
plt.quiver(X1,X2, a_tau[0].reshape((Nx,Ny)).T,a_tau[1].reshape((Nx,Ny)).T,color=colors[1], label=r'Model ($\tau=$'+str(np.round(stride*dt,2))+')')
plt.quiver(X1,X2, a_vals[0],a_vals[1],color=colors[2], label=r'Model ($\tau=0$)')
plt.title('Diffusion')
plt.xlabel('$x_1$', fontsize = 24)
plt.ylabel('$x_2$', fontsize = 24)
plt.legend(fontsize=14,loc=(1.05,0.57))
plt.grid()

plt.subplots_adjust(wspace=0.3)
plt.show()

# %%
# streamplot
plt.streamplot(X1,X2,f_vals[0],f_vals[1],color='k')

# %% 
# plot actual trajectories vs model trajectories
inits_idx = np.random.choice(np.arange(num_loc),size=20)
inits = X_t_high[inits_idx,0,:2].T

# %%
# phase portrait
xvec = np.linspace(-30,20,50)
yvec = np.linspace(-30,15,50)
fig = pplane.plot_portrait(f1_high,f2_high,xvec,yvec)#,ICs=[inits[:,i] for i in range(len(inits_idx))],tVec=np.linspace(0,dt*t_change,100))

# %%
traj_model = stochastic_sim_EM(inits, lambda x: f_high(x[0],x[1]), lambda x: sigma_high(x[0],x[1]), t_change, dt)
for i in range(len(inits_idx)):
    plt.plot(traj_model[0,:,i],traj_model[1,:,i],'k-',alpha=0.5)
    plt.plot(X_t_high[inits_idx[i],:,0],X_t_high[inits_idx[i],:,1],'r-',alpha=0.25)

plt.ylim([-10,10])
plt.xlim([-5,10])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['Stochastic Simulation (Model)','Data'])

# %%
#### Potential U, vector field decomposition

xvec = np.linspace(-30,20,100)
yvec = np.linspace(-15,15,100)
centers_new = [xvec,yvec]

X1,X2 = np.meshgrid(xvec,yvec)
f_vals_new = f_high(X1,X2)
a_vals_new = a_high(X1,X2)

# %%
# Potential U, vector field decomposition
U, grad_term, flux_term = gp.grad_flux_decomposition(np.swapaxes(f_vals_new,1,2),np.swapaxes(a_vals_new,1,2),centers_new)

#grad = grad_term.copy()
# normalize gradient
grad = grad_term/np.sqrt(grad_term[0]**2+grad_term[1]**2)

#flux = flux_term.copy()
# normalize flux
flux = flux_term/np.sqrt(flux_term[0]**2+flux_term[1]**2)

# %%
fig, ax = plt.subplots()

# plot potential and gradient/flux fields
im = ax.imshow(U.T,interpolation='nearest', origin='lower',
           extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
           cmap='jet', aspect=(xvec[-1]-xvec[0])/(yvec[-1]-yvec[0]))

downsample=8
plt.quiver(xvec[::downsample],yvec[::downsample],grad[0][::downsample,::downsample].T,grad[1][::downsample,::downsample].T,color='w',pivot='tail')
plt.quiver(xvec[::downsample],yvec[::downsample],flux[0][::downsample,::downsample].T,flux[1][::downsample,::downsample].T,color='m',pivot='tail')
plt.xlabel('PC1')
plt.ylabel('PC2')
fig.colorbar(im,label='$-\ln P$')

# %%

# pad for extended PC2 grid - having numerical trouble computing U over extended ylim
num_pad = int(1 + (15+30)/(yvec[1]-yvec[0]))
mat_padding = U.max()*np.ones((len(xvec),num_pad-len(yvec)))
U_pad = np.concatenate((mat_padding,U),axis=1)
y_pad = np.linspace(-30,15,num_pad)

# %%
X1, X2 = np.meshgrid(xvec,y_pad)
a_vals_pad = a_high(X1,X2)
f_vals_pad = f_high(X1,X2)
grad_pad = gp.gradient_flow_term(U_pad,np.swapaxes(a_vals_pad,1,2),[xvec,y_pad],isConstant=False)
flux_pad = np.swapaxes(f_vals_pad,1,2) - grad_pad

grad_pad = grad_pad/np.sqrt(grad_pad[0]**2+grad_pad[1]**2)

#flux = flux_term.copy()
# normalize flux
flux_pad = flux_pad/np.sqrt(flux_pad[0]**2+flux_pad[1]**2)
# %%
fig, ax = plt.subplots()
# plot potential and gradient/flux fields
im = ax.imshow(U_pad.T,interpolation='nearest', origin='lower',
           extent=[xvec[0], xvec[-1], y_pad[0], y_pad[-1]],
           cmap='jet', aspect=(xvec[-1]-xvec[0])/(y_pad[-1]-y_pad[0]))

downsample=8
plt.quiver(xvec[::downsample],y_pad[::downsample],grad_pad[0][::downsample,::downsample].T,grad_pad[1][::downsample,::downsample].T,color='w',pivot='tail')
plt.quiver(xvec[::downsample],y_pad[::downsample],flux_pad[0][::downsample,::downsample].T,flux_pad[1][::downsample,::downsample].T,color='m',pivot='tail')
plt.xlabel('PC1')
plt.ylabel('PC2')
fig.colorbar(im,label='$-\ln P$')

# %%
# plot U as a 3D surface
X1, X2 = np.meshgrid(xvec,y_pad,indexing='ij')
fig = plt.figure(figsize=plt.figaspect(1/3))
ax1 = fig.add_subplot(1,2,1, projection='3d')
surf = ax1.plot_surface(X1,X2, U_pad, cmap='jet')
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
