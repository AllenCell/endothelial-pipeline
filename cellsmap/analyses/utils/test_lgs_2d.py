import sys
import numpy as np
sys.path.append('//allen/aics/assay-dev/users/Erin/git-repos/cellsmap/cellsmap/analyses/utils/langevin_sindy')
import fp_solvers as fps
import langevin_sindy as lg
from stochastic_sim import stochastic_sim_EM
import sympy
from time import time
import torch


logfile="../logs/2d_test.txt"

with open(logfile, 'w') as f:
    print("GPU available: "+str(torch.cuda.is_available()), file=f)
    print("    Device: "+str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))+"\n", file=f)

def f(x1,x2):
    return np.array([(x1 + 5)*(x2 - 4), (x2 + 2)*(x1 - 3)])

def sigma(x1,x2):
    return np.array([0.5*x1 + 0.01,0.075*x2 + 0.01])

def a(x1,x2):
    return 0.5*sigma(x1,x2)**2

Nx=30
Ny=20
bins = [np.linspace(-10,10,Nx+1),np.linspace(-10,10,Ny+1)]
dx = [bins[0][1]-bins[0][0], bins[1][1]-bins[1][0]]

centers = [0.5*(bins[0][1:]+bins[0][:-1]),0.5*(bins[1][1:]+bins[1][:-1])]

X1,X2=np.meshgrid(centers[0],centers[1])

inits = np.random.uniform(low = [-7,-7], high = [7,7], size = (1000,2)).T
dt = 0.01
trajs = stochastic_sim_EM(inits, lambda x: f(x[0],x[1]), lambda x: sigma(x[0],x[1]), dt = dt, n_timepoints = 5000)

stride = 5
X_t = np.swapaxes(trajs,0,2)

num_traj = X_t.shape[0]
num_t = X_t.shape[1]
num_feats = X_t.shape[2]

data = []
data_stationary = []
for j in range(num_traj):
    my_idx = []
    idx_stat = []
    for i in range(num_t):
        my_cond = (bins[0][0] <= X_t[j,i,0] <= bins[0][-1]) and (bins[1][0] <= X_t[j,i,1] <= bins[1][-1])
        if my_cond:
            my_idx.append(i) 
    if len(my_idx) > 0:
        idx_stat = [i for i in my_idx if i >= num_t-200]
        data.append(X_t[j,my_idx,:])
    if len(idx_stat) > 0:
        data_stationary.append(X_t[j,idx_stat,:])

p_hist = np.histogram2d(np.concatenate(data_stationary)[:,0],np.concatenate(data_stationary)[:,1],bins,density=True)[0]
np.save('../outputs/bins_test.npy',np.array(bins,dtype=object),allow_pickle=True)
np.save('../outputs/p_hist_test.npy',p_hist)

### Build SINDy libraries with sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')

nf=2
ns=1
f_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(nf+1) for k in range(m+1)]),2)  # Polynomial library for drift
s_expr = np.tile(np.array([(x1**k)*(x2**(m-k)) for m in range(ns+1) for k in range(m+1)]),2)  # Polynomial library for diffusion

# Convert sympy expressions into library matrices
lib_f1 = np.zeros([len(f_expr)//2,Nx,Ny])
for k in range(len(f_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], f_expr[k])
    for i in range(Nx):
        for j in range(Ny):
            lib_f1[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

lib_f2 = np.zeros([len(f_expr)//2,Nx,Ny])
for k in range(len(f_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], f_expr[k+len(f_expr)//2])
    for i in range(Nx):
        for j in range(Ny):
            lib_f2[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

lib_f1 = lib_f1.T.reshape(Nx*Ny,-1)
lib_f2 = lib_f2.T.reshape(Nx*Ny,-1)

lib_f = np.block([[lib_f1, np.zeros((Nx*Ny,len(f_expr)//2))], [np.zeros((Nx*Ny,len(f_expr)//2)),lib_f2]])

lib_s1 = np.zeros([len(s_expr)//2,Nx,Ny])
for k in range(len(s_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], s_expr[k])
    for i in range(Nx):
        for j in range(Ny):
            lib_s1[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

lib_s2 = np.zeros([len(s_expr)//2,Nx,Ny])
for k in range(len(s_expr)//2):
    lamb_expr = sympy.lambdify([x1,x2], s_expr[k+len(s_expr)//2])
    for i in range(Nx):
        for j in range(Ny):
            lib_s2[k,i,j] = lamb_expr(X1[j,i],X2[j,i])

lib_s1 = lib_s1.T.reshape(Nx*Ny,-1)
lib_s2 = lib_s2.T.reshape(Nx*Ny,-1)

lib_s = np.block([[lib_s1, np.zeros((Nx*Ny,len(s_expr)//2))], [np.zeros((Nx*Ny,len(s_expr)//2)),lib_s2]])

f_KM, a_KM, f_err, a_err = lg.KM_avg_2D(data, bins, stride=stride, dt=dt, multi_traj=True)
np.save('../outputs/KM_drift_test.npy',f_KM)
np.save('../outputs/KM_diff_test.npy',a_KM)
np.save('../outputs/KM_drift_err_test.npy',f_err)
np.save('../outputs/KM_diff_err_test.npy',a_err)

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
W = np.array((f_err.reshape((Nx*Ny,2)), a_err.reshape(Nx*Ny,2)))
W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large numbers (small weights)
W[np.logical_not(np.isfinite(W))] = 1e6                 # Set NaN entries to large numbers (small weights)
W = 1/W  # Invert error for weights
W = W/np.nansum(W.flatten())

# Initialize adjoint solver
afp = fps.AdjFP(centers,ndim=2)

# Initialize forward steady-state solver
fp = fps.SteadyFP(centers,ndim=2)

# Optimization parameters
params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
          "f_expr": f_expr, "s_expr": s_expr,
          "lib_f": lib_f, "lib_s": lib_s, "N": (Nx,Ny),
          "kl_reg": 0,
          "fp": fp, "afp": afp, "p_hist": p_hist, "tau": stride*dt,
          "radial": False}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: lg.AFP_opt(lg.cost, params)

start_time = time()
with open(logfile, 'a') as f:
    print("Optimizing... \n", file=f)

Xi, V = lg.SSR_loop(opt_fun, params)

with open(logfile, 'a') as f:
    print("Full optimization took "+str(time()-start_time)+" seconds \n",file=f)

coeff_file = '../outputs/coeffs_test.npy'
cost_file = '../outputs/cost_test.npy'

with open(logfile, 'a') as f:
    print("Saving results to " + coeff_file + " and " + cost_file + "\n",file=f)

np.save(coeff_file,Xi)
np.save(cost_file,V)


