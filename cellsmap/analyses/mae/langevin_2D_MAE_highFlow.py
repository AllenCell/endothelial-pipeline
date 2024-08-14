import numpy as np
import sys
sys.path.append('//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/utils/langevin_sindy')
import sympy
import fp_solvers as fps
import langevin_sindy as lg
import torch
from time import time

logfile="../logs/2d_mae_highFlow.txt"

with open(logfile, 'w') as f:
    print("GPU available: "+str(torch.cuda.is_available()), file=f)
    print("    Device: "+str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))+"\n", file=f)


stride = 9
dt = 5
X_t_high = np.load('../data/MAE_95pctVarPCs_highFlow.npy')

num_loc = X_t_high.shape[0]
num_feats = X_t_high.shape[2]

data = [X_t_high[i,:,:2] for i in range(num_loc)] # high flow data, pass as list into KM_avg
data_stationary = [X_t_high[i,100:,:] for i in range(num_loc)] # "Steady state" low flow data, for histogram

N = 32
min0 = min([min(traj[:,0]) for traj in data])
max0 = max([max(traj[:,0]) for traj in data])
bin0_min = 0.5*(np.floor(min0)+np.round(min0,1))
bin0_max = 0.5*(np.ceil(max0)+np.round(max0,1))
bins0 = np.linspace(bin0_min, bin0_max, N+1)
centers0 = 0.5*(bins0[1:]+bins0[:-1])


min1 = min([min(traj[:,1]) for traj in data])
max1 = max([max(traj[:,1]) for traj in data])
bin1_min = 0.5*(np.floor(min1)+np.round(min1,1))
bin1_max = 0.5*(np.ceil(max1)+np.round(max1,1))
bins1 = np.linspace(bin1_min, bin1_max, N+1)
centers1 = 0.5*(bins1[1:]+bins1[:-1])

dx = [bins0[1]-bins0[0],bins1[1]-bins1[0]]

bins = [bins0,bins1]
centers = [centers0,centers1]

p_hist, _, _ = np.histogram2d(np.concatenate(data_stationary)[:,0],np.concatenate(data_stationary)[:,1], bins, density=True)
np.save('../outputs/bins_mae_highFlow.npy',bins)
np.save('../outputs/p_hist_mae_highFlow.npy',p_hist)

## KM average (coarse grained subsampling)
f_KM, a_KM, f_err, a_err = lg.KM_avg_2D(data, bins, stride=stride, dt=dt, multi_traj=True)
np.save('../outputs/KM_drift_mae_highFlow.npy',f_KM)
np.save('../outputs/KM_diff_mae_highFlow.npy',a_KM)
np.save('../outputs/KM_drift_err_mae_highFlow.npy',f_err)
np.save('../outputs/KM_diff_err_mae_highFlow.npy',a_err)

### Build SINDy libraries with sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')

nf=3
ns=2
f_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(nf+1) for k in range(m+1)]),2)  # Polynomial library for drift
s_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(ns+1) for k in range(m+1)]),2)  # Polynomial library for diffusion

c0,c1 = np.meshgrid(centers0,centers1,indexing='ij')
# Convert sympy expressions into library matrices
lib_f1 = np.zeros([len(f_expr)//2,N,N])
for k in range(len(f_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], f_expr[k])
    for i in range(N):
        for j in range(N):
            lib_f1[k,i,j] = lamb_expr(c0[i,j],c1[i,j])

lib_f2 = np.zeros([len(f_expr)//2,N,N])
for k in range(len(f_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], f_expr[k+len(f_expr)//2])
    for i in range(N):
        for j in range(N):
            lib_f2[k,i,j] = lamb_expr(c0[i,j],c1[i,j])

lib_f1 = lib_f1.T.reshape(N**2,-1)
lib_f2 = lib_f2.T.reshape(N**2,-1)

lib_f = np.block([[lib_f1, np.zeros((N**2,len(f_expr)//2))], [np.zeros((N**2,len(f_expr)//2)),lib_f2]])

lib_s1 = np.zeros([len(s_expr)//2,N,N])
for k in range(len(s_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], s_expr[k])
    for i in range(N):
        for j in range(N):
            lib_s1[k,i,j] = lamb_expr(c0[i,j],c1[i,j])

lib_s2 = np.zeros([len(s_expr)//2,N,N])
for k in range(len(s_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], s_expr[k+len(s_expr)//2])
    for i in range(N):
        for j in range(N):
            lib_s2[k,i,j] = lamb_expr(c0[i,j],c1[i,j])

lib_s1 = lib_s1.T.reshape(N**2,-1)
lib_s2 = lib_s2.T.reshape(N**2,-1)

lib_s = np.block([[lib_s1, np.zeros((N**2,len(s_expr)//2))], [np.zeros((N**2,len(s_expr)//2)),lib_s2]])

# Initialize Xi with least squares regression (no finite-time corrections)

m=len(f_expr)+len(s_expr)
Xi0 = np.zeros(m)
mask = (np.where(np.isfinite(f_KM[:,:,0].flatten())*np.isfinite(f_KM[:,:,1].flatten())))[0]
n_mask = len(mask)
A1 = np.block([[lib_f1[mask], np.zeros((n_mask,len(f_expr)//2))], [np.zeros((n_mask,len(f_expr)//2)),lib_f2[mask]]])
b1 = np.hstack((f_KM[:,:,0].flatten()[mask],f_KM[:,:,1].flatten()[mask])).T
Xi0[:len(f_expr)] = np.linalg.lstsq(A1, b1, rcond=None)[0]   # Regression against drift

mask = (np.where(np.isfinite(a_KM[:,:,0].flatten())*np.isfinite(a_KM[:,:,1].flatten())))[0]
n_mask = len(mask)
A2 = np.block([[lib_s1[mask], np.zeros((n_mask,len(s_expr)//2))], [np.zeros((n_mask,len(s_expr)//2)),lib_s2[mask]]])
b2 = np.hstack((a_KM[:,:,0].flatten()[mask],a_KM[:,:,1].flatten()[mask])).T
Xi0[len(f_expr):] = np.linalg.lstsq(A2,b2, rcond=None)[0]  # Regression against diffusion a

### Weights: uncertainties in Kramers-Moyal
# This is helpful, but not that critical.  The specific choice of weights doesn't matter that much
W = np.array((f_err.reshape((N*N,2)), a_err.reshape(N*N,2)))
W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large numbers (small weights)
W[np.logical_not(np.isfinite(W))] = 1e6                 # Set NaN entries to large numbers (small weights)
W = 1/W  # Invert error for weights
W = W/np.nansum(W.flatten())

# Initialize adjoint solver
centers = [centers0,centers1]
afp = fps.AdjFP(centers,ndim=2)

# Initialize forward steady-state solver
fp = fps.SteadyFP((N,N), dx)

# Optimization parameters
params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
          "f_expr": f_expr, "s_expr": s_expr,
          "lib_f": lib_f, "lib_s": lib_s, "N": (N,N),
          "kl_reg": 0.5,
          "fp": fp, "afp": afp, "p_hist": p_hist, "tau": stride*dt,
          "radial": False}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: lg.AFP_opt(lg.cost, params)
start_time = time()
with open(logfile, 'a') as f:
    print("Optimizing... \n",file=f)

Xi, V = lg.SSR_loop(opt_fun, params)
with open(logfile, 'a') as f:
    print("Full optimization took "+str(time()-start_time)+" seconds \n",file=f)

coeff_file = '../outputs/coeffs_mae_highFlow.npy'
cost_file = '../outputs/cost_mae_highFlow.npy'

# Save the results
with open(logfile, 'a') as f:
    print("Saving results to " + coeff_file + " and " + cost_file + "\n",file=f)

np.save(coeff_file,Xi)
np.save(cost_file,V)
