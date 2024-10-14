# %%
import cellsmap.analyses.workflows.analyze_SDE_model as asm
from cellsmap.util import io
import numpy as np
from cellsmap.analyses.utils import numerical_fplanck as fplanck
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
mpl.rcParams['animation.embed_limit'] = 2**128
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

# %%
config_name = 'mae_cdh5_small_patch_LR'
config_inputs = io.get_dynamics_inputs(config_name)
savedir = config_inputs[-2]
ndim = config_inputs[2]
split_high_low = config_inputs[6]
split_frame = config_inputs[7][0][-1]
X_t = np.load(savedir + 'data/traj_array.npy')

flow = "low"
n_terms = 14
X_t_low = X_t[:,split_frame:,:]


# %%
V, Xi, f_expr, s_expr = asm.load_model_outputs(savedir,flow)

f, D, _ = asm.get_model_functions(n_terms, ndim, V, Xi, f_expr, s_expr)

def D_const(x1,x2):
    d1 = D(0,0)[0]
    d2 = D(0,0)[1]
    if np.isscalar(d1):
        d1 = d1 + 0*x1
    if np.isscalar(d2):
        d2 = d2 + 0*x2
    return np.array([d1,d2])

# %%
# initialize time dependent fokker planck solver (only need for grid)
sim = fplanck.fokker_planck(force=f,diffusion=D_const, extent=[[-30,30],[-30,30]],Ngrid=[35,35])

# %%
def kde_p0(X1,X2,indexing='ij'):
    data0 = X_t_low[:,0,:2].reshape(-1,2)
    kde = KernelDensity(kernel="gaussian", bandwidth=1.75).fit(data0)
    if indexing == 'ij':
        log_dens = kde.score_samples(np.array([X1.T.ravel(), X2.T.ravel()]).T)
        N1 = X1.shape[0]
        N2 = X1.shape[1]
    else:
        log_dens = kde.score_samples(np.array([X1.ravel(), X2.ravel()]).T)
        N1 = X1.shape[1]
        N2 = X1.shape[0]
    return np.exp(log_dens).reshape((N2,N1)).T

p0 = kde_p0(*sim.grid)
tol = 1e-10
p0[p0 < tol] = tol
p0 = p0/np.sum(p0)
fig, ax = plt.subplots()
ax.imshow(p0.T,interpolation='nearest', origin='lower',
           extent=[sim.axes[0][0], sim.axes[0][-1], sim.axes[1][0], sim.axes[1][-1]],
           aspect=(sim.axes[0][-1]-sim.axes[0][0])/(sim.axes[1][-1]-sim.axes[1][0]))

# %%
from scipy.linalg import expm, norm
# from scipy import sparse
from scipy.sparse.linalg import expm_multiply
Nsteps = X_t_low.shape[1]
tf = 5*(Nsteps-1)
time_vec = np.linspace(0, tf, Nsteps)
Pt = np.zeros((Nsteps,)+tuple(sim.Ngrid))
Pt[0] = p0

my_mat = sim.master_matrix

# %%
Pt = expm_multiply(my_mat, p0.flatten(), start=0, stop=tf, num=Nsteps, endpoint=True).reshape((Nsteps,)+tuple(sim.Ngrid))

# %%
for i, t in enumerate(time_vec):
    if i == 0:
        continue
    if i % 20 == 0:
        print('Time step {0}'.format(i))
    Pt[i] = expm_multiply(my_mat*t,p0.flatten()).reshape(sim.Ngrid)
# %%
np.save(savedir+'data/fokker_planck_lowFlow_sim.npy',Pt)

# %%
# Pt = np.load(savedir+'data/fokker_planck_lowFlow_sim.npy')

# %%
### animation
# change the x and ylim to clip the bounds of the plots
xlim = [sim.axes[0][1],sim.axes[0][-2]]
ylim = [sim.axes[1][1],sim.axes[1][-2]]
xvec = sim.grid[0]
yvec = sim.grid[1]

fig = plt.figure(figsize=plt.figaspect(1/2))
ax1 = fig.add_subplot(1,2,1, projection='3d')

p0_temp = np.where((xvec > xlim[1]) | (xvec < xlim[0]), None, p0)
p0_temp = np.where((yvec > ylim[1]) | (yvec < ylim[0]), None, p0_temp)
surf = ax1.plot_surface(xvec,yvec, p0_temp, cmap='viridis')

ax1.set_zlim([0,1.1*np.nanmax(Pt)])
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.autoscale(False)

ax1.set(xlabel='PC1', ylabel='PC2', zlabel='normalized PDF')

ax2 = fig.add_subplot(1,2,2)

im = ax2.pcolormesh(*sim.grid, p0[:-1,:-1], vmax=1.1*np.max(Pt),shading='flat')
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

def update(i):
    global surf
    surf.remove()
    P_temp = np.where((xvec > xlim[1]) | (xvec < xlim[0]), None, Pt[i])
    P_temp = np.where((yvec > ylim[1]) | (yvec < ylim[0]), None, P_temp)
    surf = ax1.plot_surface(xvec,yvec, P_temp, cmap='viridis')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    data = Pt[i, :-1,:-1]
    im.set_array(np.ravel(data))
    im.set_clim(vmax=1.1*np.nanmax(Pt))
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    
    return [surf, im]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=30)
plt.tight_layout()
plt.show()

# %%

HTML(anim.to_jshtml())

# %%
writer = PillowWriter(fps=30,bitrate=1800)
anim.save(savedir+'figs/lowFlow_fpe.gif', writer=writer)

# %%
