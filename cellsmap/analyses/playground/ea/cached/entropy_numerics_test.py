# %%
import numpy as np

import matplotlib.pyplot as plt
import numdifftools as nd

import cellsmap.analyses.utils.gen_potential as gp
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps
import cellsmap.analyses.utils.viz as viz

# %%
D_const = 2 
A = np.array([[-2,0],[0,-4]])
ndim = A.shape[0]

def f(x):
    return A@x

def D(x):
    if x.__class__ == np.ndarray:
        if len(x.shape) == 1:
            return D_const*np.diag(x**2)
        elif len(x.shape) == 2:
            D_mat = np.zeros((ndim,ndim,x.shape[-1]))
            for i in range(x.shape[-1]):
                D_mat[:,:,i] = D_const*np.diag(x[:,i]**2)
            return D_mat
        elif len(x.shape) == 3:
            return D(x.reshape((ndim,-1)))
    elif x.__class__ == list:
        D_mat = np.zeros((ndim,ndim,len(x[0]),len(x[1])))
        for i in range(len(x[0])):
            for j in range(len(x[1])):
                D_mat[:,:,i,j] = D_const*np.diag([x[0][i]**2,x[1][j]**2])
        return D_mat

def div_D(x):
    return 2*D_const*x

def J_flux(x,p,grad_p):
    return f(x)*p - div_D(x)*p - np.einsum('ijx,jx->ix', D(x_grid), grad_p)

def invert_D(D):
    if len(D.shape) == 2:
        return np.linalg.inv(D)
    else:
        if len(D.shape) == 4:
            ndim = D.shape[0]
            D_ = D.reshape((ndim,ndim,-1))
        elif len(D.shape) == 3:
            D_ = D.copy()
        D_inv = np.zeros(D_.shape)
        for i in range(D_.shape[-1]):
            D_inv[:,:,i] = np.linalg.inv(D_[:,:,i])
        return D_inv

def entropy_production(J,D,V,x):
    D_inv = invert_D(D)
    D_inv_J = np.einsum('ijx,jx->ix', D_inv, J)
    inner_prod = np.einsum('ij,ij->j',D_inv_J,V)
    my_integral = np.copy(inner_prod).reshape((len(x[0]),len(x[1])))
    for i in range(ndim):
        my_integral = np.trapz(my_integral,x[i],axis=0)
    return my_integral

# %%
x1 = np.linspace(0.001,0.5,50)
x2 = np.linspace(0.001,0.5,50)
x_vals = [x1,x2]
dx = [x1[1]-x1[0],x2[1]-x2[0]]
N_grid = [len(x1),len(x2)]
X1,X2 = np.meshgrid(x1,x2) 

x_grid = np.vstack([X1.flatten(),X2.flatten()])
f_vals = f(x_grid).reshape((-1,N_grid[0],N_grid[1]))
D_vals = D(x_grid).reshape((ndim,ndim,N_grid[0],N_grid[1]))

D_flat = np.zeros((ndim,N_grid[0],N_grid[1]))
for i in range(ndim):
    D_flat[i] = D_vals[i,i]
# %%
tol = 1e-8
stationary_fp = fps.SteadyFP(N_grid, dx)
P = stationary_fp.solve(f_vals,D_flat)
P[P<tol] = tol
P = P/np.sum(P)


# %%
bins = []
for i in range(ndim):
    bin_min = x_vals[i].min()-dx[i]/2
    bin_max = x_vals[i].max()+dx[i]/2
    bins.append(np.linspace(bin_min,bin_max,N_grid[i]+1))
fig,ax = viz.init_plot()
ax = viz.plot_histogram_2D(ax,P,bins,cmap='inferno') # plot empirical PDF
# %%
J = gp.probability_flux(P, f_vals, D_flat,x_vals)
grad_P = np.array(np.gradient(P,*x_vals,edge_order=2))

J_true = J_flux(x_grid,P.flatten(),grad_P.reshape((ndim,-1)))

print(np.linalg.norm(J.reshape((ndim,-1))-J_true))
# %%
plt.quiver(x1,x2, J_true[0].reshape((N_grid[0],N_grid[1])),
           J_true[1].reshape((N_grid[1],N_grid[1])),color='k')
plt.quiver(x1,x2, J[0].reshape((N_grid[0],N_grid[1])),
              J[1].reshape((N_grid[1],N_grid[1])),color='r')
# %%
D_x_grad_P = np.einsum('ijx,jx->ix', D(x_grid), grad_P.reshape((ndim,-1)))
V_true = (A-2*D_const*np.eye(2))@x_grid-D_x_grad_P/P.flatten()
V = J/P
print(np.linalg.norm(V_true-V.reshape((ndim,-1))))

# %%
plt.quiver(x1,x2, V_true[0].reshape((N_grid[0],N_grid[1])),
              V_true[1].reshape((N_grid[1],N_grid[1])),color='k')
plt.quiver(x1,x2, V[0].reshape((N_grid[0],N_grid[1])),
                V[1].reshape((N_grid[1],N_grid[1])),color='r')

# %%
num_points = [10,15,25,40,50,75]
flux_err = []
V_err = []
epr_num = []
for n in num_points:
    x1 = np.linspace(0.001,0.5,n)
    x2 = np.linspace(0.001,0.5,n)
    x_vals = [x1,x2]
    dx = [x1[1]-x1[0],x2[1]-x2[0]]
    N_grid = [n,n]
    X1,X2 = np.meshgrid(x1,x2) 
    x_grid = np.vstack([X1.flatten(),X2.flatten()])

    f_vals = f(x_grid).reshape((-1,N_grid[0],N_grid[1]))
    D_vals = D(x_grid).reshape((ndim,ndim,N_grid[0],N_grid[1]))

    D_flat = np.zeros((ndim,N_grid[0],N_grid[1]))
    for i in range(ndim):
        D_flat[i] = D_vals[i,i]
    tol=1e-8

    stationary_fp = fps.SteadyFP(N_grid, dx)
    P = stationary_fp.solve(f_vals,D_flat)
    P[P<tol] = tol
    P = P/np.sum(P)
    grad_P = np.array(np.gradient(P,*x_vals,edge_order=2))

    J = gp.probability_flux(P, f_vals, D_flat,x_vals)
    J_true = J_flux(x_grid,P.flatten(),grad_P.reshape((ndim,-1)))

    D_x_grad_P = np.einsum('ijx,jx->ix', D_vals.reshape((ndim,ndim,-1)), grad_P.reshape((ndim,-1)))
    V_true = (A-2*D_const*np.eye(2))@x_grid-D_x_grad_P/P.flatten()
    V = J/P

    flux_err.append(np.linalg.norm(J.reshape((ndim,-1))-J_true))
    V_err.append(np.linalg.norm(V.reshape((ndim,-1))-V_true))

    epr_num.append(entropy_production(J.reshape((ndim,-1)),D_vals,
                                      V.reshape((ndim,-1)),x_vals))

# %%
plt.semilogx(num_points, flux_err,'k-o')
# %%
plt.semilogx(num_points, V_err,'k-o')
# %%
plt.plot(num_points, epr_num,'k-o')
# %%
