# %%
import numpy as np
import matplotlib.pyplot as plt
import cellsmap.analyses.utils.langevin_sindy.fp_solvers as fps
import cellsmap.analyses.utils.cached.gen_potential as gp
import cellsmap.analyses.utils.pplane as pp
import numdifftools as nd
import cellsmap.analyses.utils.viz as viz

# %%
# Wilson-Cowan model response function
def R(x,b,theta):
    return 1/(1+np.exp(-b*(x-theta))) - 1/(1+np.exp(b*theta))

# %%
# parameters for excitatory and inhibitory subpopulations
#
# from Jin Wang's paper "Thermodynamic and dynamical predictions 
#                           for bifurcations and non-equilibrium phase transitions"
#
global W_ee, W_ie, b_e, theta_e, k_e, R_e, W_ei, W_ii, b_i, theta_i, k_i, I_i, R_i, D

W_ee = 16
W_ie = 12
b_e = 1.3
theta_e = 4
k_e = 0.85
R_e = lambda x: R(x,b_e,theta_e)

W_ei = 4
W_ii = 3
b_i = 2
theta_i = 3.7
k_i = 0.85
I_i = 0
R_i = lambda x: R(x,b_i,theta_i)

# diffusion coefficient (finite noise case)
D = 0.0032
D_func = lambda x: np.array([D,D])
# %%
def f(x,I_e):
    '''ODE model dx/dt = f(x) for the Wilson-Cowan model'''
    x_e = x[0]
    x_i = x[1]

    dx_e = -x_e + (k_e - x_e)*R_e(W_ee*x_e - W_ie*x_i + I_e)
    dx_i = -x_i + (k_i - x_i)*R_i(W_ei*x_e - W_ii*x_i + I_i)

    return np.array([dx_e,dx_i])
# %%
# check the fixed points of the Wilson-Cowan model for different values of I_e

I_e_vec = np.linspace(-1.9,2.5,60)
x_fp = np.zeros((2,len(I_e_vec)))

x1_lims = [-0.3,0.6]
x2_lims = [-0.2,0.2]

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],10)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],10)

fpt_dict = {}

for (ii, I_e) in enumerate(I_e_vec):
    def myFlow(x):
            return f(x,I_e=I_e)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [np.array([x1_coarse[i],x2_coarse[j]]) for i in range(len(x1_coarse)) for j in range(len(x2_coarse))]
    fpts = pp.get_fps(myFlow,init_coarse) # get fixed points
    fpt_types = []
    if len(fpts) > 0:
        for fpt in fpts:
            fptStability = pp.find_stability(flowJacobian(fpt))
            if 'Stable' in fptStability:
                fpt_types.append('stable')
            elif 'Unstable' in fptStability:
                fpt_types.append('unstable')
            elif 'Saddle' in fptStability:
                fpt_types.append('saddle')
            else:
                fpt_types.append('indeterminate')
    fpts_new = []
    fpt_types_new = []
    for fpt in fpts:
        # if far out of bounds of the plot window, don't report it
        if fpt[0]<x1[0]-0.5*abs(x1[0]) or fpt[0]>x1[-1]+0.5*abs(x1[-1]) or fpt[1]<x2[0]-0.5*abs(x2[0]) or fpt[1]>x2[-1]+0.5*abs(x2[-1]):
            continue
        else:
            fpts_new.append(fpt)
            fpt_types_new.append(fpt_types[fpts.index(fpt)])

    fpt_dict[str(I_e)] = {}
    fpt_dict[str(I_e)]['fixed_points'] = fpts_new
    fpt_dict[str(I_e)]['fixed_point_types'] = fpt_types_new
# %%
for I_e in I_e_vec:
    fpts = fpt_dict[str(I_e)]['fixed_points']
    fpt_types = fpt_dict[str(I_e)]['fixed_point_types']
    if len(fpts) > 0:
        for i,fpt in enumerate(fpts):
            if fpt_types[i] == 'stable':
                color = 'b'
            elif fpt_types[i] == 'unstable':
                color = 'r'
            elif fpt_types[i] == 'saddle':
                color = 'tab:purple'
            else:
                color = 'darkgoldenrod'

            plt.plot(I_e,fpt[0],'o',color=color)
plt.xlabel('$I_E$')
plt.ylabel('$x_E^*$')


# %%
I_e_vec = np.linspace(-1.9,2.5,60)

XE = np.linspace(-0.3,0.6,50)
XI = np.linspace(-0.2,0.2,40)
my_mesh = np.meshgrid(XE,XI)

# %%
f_vals = np.zeros((2,XE.shape[0],XI.shape[0],len(I_e_vec)))
D_vals = np.zeros((2,XE.shape[0],XI.shape[0],len(I_e_vec)))

tol = 1e-6

N_grid = [XE.shape[0],XI.shape[0]]
dx = [XE[1]-XE[0],XI[1]-XI[0]]

# entropy production rate as a function of u
epr = np.zeros(len(I_e_vec))
for (ii, I_e) in enumerate(I_e_vec):
    for i in range(XE.shape[0]):
        for j in range(XI.shape[0]):
            x = np.array([my_mesh[0][j,i],my_mesh[1][j,i]])
            f_vals[:,i,j,ii] = f(x,I_e=I_e)
            D_vals[:,i,j,ii] = D_func(x)
    

    fp = fps.SteadyFP(N_grid, dx)

    P = fp.solve(f_vals[:,:,:,ii],D_vals[:,:,:,ii]) # solve stationary Fokker-Planck equation
    P[P<tol] = tol # set small values to a small number to avoid numerical issues

    J = gp.probability_flux(P,f_vals[:,:,:,ii],D_vals[:,:,:,ii],[XE,XI])

    if ii == 0:
        fig, ax = plt.subplots()
        bins = [np.linspace(XE[0]-0.5*dx[0],XE[-1]+0.5*dx[0],N_grid[0]+1),
                np.linspace(XI[0]-0.5*dx[1],XI[-1]+0.5*dx[1],N_grid[1]+1)]
        ax = viz.plot_histogram_2D(ax,P,bins,cmap='inferno') # plot model PDF

        fig,ax = plt.subplots()
        ax.quiver(my_mesh[0],my_mesh[1],J[0].T,J[1].T,color='k')

    D_mat = gp.expand_to_matrix(D_vals[:,:,:,ii])
    epr[ii] = gp.entropy_production(J, D_mat, P, [XE,XI])

# %%
plt.plot(I_e_vec,epr,'-o',color='k')
# %%
# marginal distribution over x_E
P_marginal = np.sum(P,axis=1)
P_marginal = P_marginal/np.sum(P_marginal)
plt.plot(XE,P_marginal)

# %%
fig, ax = plt.subplots()
bins = [np.linspace(XE[0]-0.5*dx[0],XE[-1]+0.5*dx[0],N_grid[0]+1),
        np.linspace(XI[0]-0.5*dx[1],XI[-1]+0.5*dx[1],N_grid[1]+1)]
ax = viz.plot_histogram_2D(ax,P,bins,cmap='inferno') # plot model PDF
# %%
