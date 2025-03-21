# %%
import numpy as np
import numdifftools as nd

from cellsmap.analyses.utils import model_analysis, model_eval, regression_helper as rh
from cellsmap.analyses.utils.numerics import gen_potential as gp
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.viz import pplane, dynamics_viz, viz_base as vb

from cellsmap.analyses.configs.manifest_postproc_config import savedir, ds_to_skip, PCs
from cellsmap.analyses.configs.dynamics_viz_config import Nbins_plot, bin_limits, plt_args, shear_range

# %%
# load fitted drift-diffusion model (list of fit SINDy objects)
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.pkl')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
myModel = [driftModel,diffModel]
# %%
# for plotting phase plane and histogram plots, fix grid and bin limits across all datasets
bins, centers = rh.get_bins(Nbins_plot,bin_limits=bin_limits)

# add keys for labeling axes in plots according to the PCs we are interested in
plt_args['plt_xlabel'] = 'PC'+str(PCs[0]+1)
plt_args['plt_ylabel'] = 'PC'+str(PCs[1]+1)

# %%
# run comparison of model and data for each dataset
model_analysis.model_data_comparison(myModel,savedir,PCs,bins,centers,ds_to_skip,plt_args)


# %%
# WRAP THESE INTO FUNCTIONS
# fixed point analysis: plot coordinates of fixed points as a function of shear stress
u_range = np.linspace(4,30,60)

fpt_dict = {}

x1_lims = plt_args['pplane_xlim']
x2_lims = plt_args['pplane_ylim']

x1 = np.linspace(x1_lims[0],x1_lims[1],50)
x2 = np.linspace(x2_lims[0],x2_lims[1],50)
x1_coarse = np.linspace(x1_lims[0],x1_lims[1],7)
x2_coarse = np.linspace(x2_lims[0],x2_lims[1],7)

f = model_eval.vector_field_function(driftModel)
for u in u_range:

    def myFlow(x):
        return f(x,u=u)
    flowJacobian = nd.Jacobian(myFlow)

    init_coarse = [np.array([x1_coarse[i],x2_coarse[j]]) 
                   for i in range(len(x1_coarse)) 
                   for j in range(len(x2_coarse))
                   ]
    fpts = pplane.get_fps(myFlow,init_coarse) # get fixed points
    fpt_types = []
    if len(fpts) > 0:
        for fpt in fpts:
            fptStability = pplane.find_stability(flowJacobian(fpt))
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

    fpt_dict[str(u)] = {}
    fpt_dict[str(u)]['fixed_points'] = fpts_new
    fpt_dict[str(u)]['fixed_point_types'] = fpt_types_new

fpt_stable = []
u_stable = []
for j in range(2):
    fig, ax = vb.init_plot()
    for u in u_range:
        if str(u) in fpt_dict.keys():
            fpts = fpt_dict[str(u)]['fixed_points']
            fpt_types = fpt_dict[str(u)]['fixed_point_types']
            if len(fpts) > 0:
                for i,fpt in enumerate(fpts):
                    if fpt_types[i] == 'stable':
                        color = 'b'
                        fpt_stable.append(fpt)
                        u_stable.append(u)
                    elif fpt_types[i] == 'unstable':
                        color = 'r'
                    elif fpt_types[i] == 'saddle':
                        color = 'tab:purple'
                    else:
                        color = 'darkgoldenrod'

                    ax.plot(u,fpt[j],'o',color=color)
                    ax.set_xlabel('Shear stress (dyn/cm^2)')
                    ax.set_ylabel('PC'+str(PCs[j]+1))
    ax.set_title('Fixed points by shear stress')
    if j == 0:
        ax.set_ylim(x1_lims)
    else:    
        ax.set_ylim(x2_lims)

# %%
# entropy production rate as a function of shear stress

model_analysis.get_epr(myModel,bins,centers,shear_range,savedir)


# %%
##### WRAP THIS INTO FUNCTION in thermo_viz.py #####
################### Generalized potential energy landscape ###################

# this should all go in config file
N_fine = 60

x_fine = np.linspace(-4,4.5,N_fine+1)
y_fine = np.linspace(-3,3.5,N_fine+1)
bins_fine = [x_fine,y_fine]
centers_fine = [0.5*(x_fine[1:]+x_fine[:-1]),
                0.5*(y_fine[1:]+y_fine[:-1])]
X1,X2 = np.meshgrid(centers_fine[0],centers_fine[1])
# %%
# specify shear stress in config file, plot this multiple times for different shear stresses
# will be a coarser list than shear_range
u = 5.5 # shear stress

tol = 1e-6

f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)

f_mesh = model_eval.mesh_grid_function(f)
D_mesh = model_eval.mesh_grid_function(D)

f_vals_new = f_mesh([X1,X2],u).T
D_vals_new = D_mesh([X1,X2],u).T

p_fit = model_eval.get_stationary_probability(f,D,bins_fine,centers_fine,u,tol=tol)
U= -np.log(p_fit)

print('**** Plotting generalized potential energy landscape **** \n')

fig,ax = dynamics_viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],cmap='jet',surf=False)
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))
ax.set_title('Shear stress: '+str(u)+' dyn/cm$^2$')

# %%
# same but with vector field decomposition
# write this as an option in the function
normed = False # if True, normalize vectors by their magnitudes

_, grad_term, _, flux_term = gp.grad_flux_decomposition(f_vals_new,D_vals_new,centers_fine,tol=tol)
grad_ = grad_term.copy()
flux_ = np.array(flux_term)
if normed:
    grad_ = grad_/(np.sqrt(grad_[0]**2+grad_[1]**2))
    flux_ = flux_/(np.sqrt(flux_[0]**2+flux_[1]**2))

fig,ax = dynamics_viz.plot_gen_potential_2D(U,centers_fine[0],centers_fine[1],
                                   cmap='jet',surf=False)
downsample=10
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          grad_[0][::downsample,::downsample].T,
          grad_[1][::downsample,::downsample].T,
          color='w',pivot='tail')
ax.quiver(centers_fine[0][::downsample],
          centers_fine[1][::downsample],
          flux_[0][::downsample,::downsample].T,
          flux_[1][::downsample,::downsample].T,
          color='r',pivot='tail')
ax.set_xlabel('PC'+str(PCs[0]+1))
ax.set_ylabel('PC'+str(PCs[1]+1))

# %%
