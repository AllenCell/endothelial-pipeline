import numpy as np
import sys
import sympy
import langevin_sindy as lg

stride = 9
dt = 5
X_t_high = np.load('MAE_95pctVarPCs_highFlow.npy')

num_loc = X_t_high.shape[0]
num_feats = X_t_high.shape[2]

data = [X_t_high[i,:,:2] for i in range(num_loc)] # high flow data, pass as list into KM_avg

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

p_hist, _, _ = np.histogram2d(np.concatenate(data)[:,0],np.concatenate(data)[:,1], bins, density=True)

## KM average (coarse grained subsampling)
f_KM, a_KM, f_err, a_err = lg.KM_avg_2D(data, bins, stride=stride, dt=dt, multi_traj=True)

### Build SINDy libraries with sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')

f_expr = np.array([(x1**i)*(x2**j) for i in np.arange(4) for j in np.arange(4)])  # Polynomial library for drift
s_expr = np.array([(x1**i)*(x2**j) for i in np.arange(3) for j in np.arange(3)])  # Polynomial library for diffusion

# Convert sympy expressions into library matrices
lib_f = np.zeros([len(f_expr),N,N])
for k in range(len(f_expr)):
    lamb_expr = sympy.lambdify([x1,x2], f_expr[k])
    lib_f[k] = lamb_expr(centers0,centers1)

lib_f = lib_f.T

lib_s = np.zeros([len(s_expr),N,N])
for k in range(len(s_expr)):
    lamb_expr = sympy.lambdify([x1,x2], s_expr[k])
    lib_s[k] = lamb_expr(centers0,centers1)

lib_s = lib_s.T

# Initialize Xi with least squares regression (no finite-time corrections)

m=len(f_expr)+len(s_expr)
Xi0 = np.zeros((m,2))
mask = np.where(np.isfinite(f_KM[:,:,0])*np.isfinite(f_KM[:,:,1]))
Xi0[:len(f_expr)] = np.linalg.lstsq( lib_f[mask[0],mask[1]], f_KM[mask[0],mask[1]], rcond=None)[0]   # Regression against drift f1
Xi0[len(f_expr):] = np.linalg.lstsq( lib_s[mask[0],mask[1]], np.sqrt(2*a_KM[mask[0],mask[1]]), rcond=None)[0]  # Regression against diffusion a1

### Weights: uncertainties in Kramers-Moyal
# This is helpful, but not that critical.  The specific choice of weights doesn't matter that much
W = np.array((f_err.flatten(), a_err.flatten()))
W[np.less(abs(W), 1e-12, where=np.isfinite(W))] = 1e6  # Set zero entries to large weights
W[np.logical_not(np.isfinite(W))] = 1e6                 # Set NaN entries to large numbers (small weights)
W = 1/W  # Invert error for weights
W = W/np.nansum(W.flatten())

# Compute empirical PDF
p_hist, _, _ = np.histogram2d(np.concatenate(data)[:,0],np.concatenate(data)[:,1], bins, density=True)

# Initialize adjoint solver
centers = np.vstack([centers0,centers1])
afp = lg.AdjFP(centers,ndim=2)

# Initialize forward steady-state solver
fp = lg.SteadyFP((N,N), dx)

# Optimization parameters
params = {"W": W, "f_KM": f_KM, "a_KM": a_KM, "Xi0": Xi0,
          "f_expr": f_expr, "s_expr": s_expr,
          "lib_f": lib_f, "lib_s": lib_s, "N": (N,N),
          "kl_reg": 10,
          "fp": fp, "afp": afp, "p_hist": p_hist, "tau": stride*dt,
          "radial": False}

# Use anonymous function to automatically pass the cost function
opt_fun = lambda params: lg.AFP_opt(lg.cost2, params)
Xi, V = lg.SSR_loop(opt_fun, params)

# Save the results
np.save('coeffs_highFlow.npy',Xi)
np.save('cost_highFlow.npy',V)
