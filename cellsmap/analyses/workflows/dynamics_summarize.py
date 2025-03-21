# %%
import numpy as np
import numdifftools as nd

from cellsmap.analyses.utils import model_analysis, model_eval, regression_helper as rh
from cellsmap.analyses.utils.numerics import gen_potential as gp
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.viz import dynamics_viz, viz_base as vb

from cellsmap.analyses.configs.manifest_postproc_config import savedir, ds_to_skip, PCs
from cellsmap.analyses.configs.dynamics_viz_config import Nbins_plot, bin_limits, \
    pplane_args, shear_range, fpt_args, Nbins_gp, shear_range_gp, p_tol, gp_args

# %%
# load fitted drift-diffusion model (list of fit SINDy objects)
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.pkl')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
myModel = [driftModel,diffModel]
# %%
# for plotting phase plane and histogram plots, fix grid and bin limits across all datasets
bins, centers = rh.get_bins(Nbins_plot,bin_limits=bin_limits)

# %%
# run comparison of model and data for each dataset
model_analysis.model_data_comparison(myModel,savedir,PCs,bins,centers,ds_to_skip,pplane_args)

# %%
# fixed point analysis: plot coordinates of fixed points as a function of shear stress
model_analysis.run_fixed_point_analysis(driftModel,shear_range,fpt_args,savedir)

# %%
# entropy production rate as a function of shear stress
model_analysis.run_epr_analysis(myModel,bins,centers,shear_range,savedir)

# %%
################### Generalized potential energy landscape ###################

# this will all go in wrapper function in model_analysis.py

bins_gp, centers_gp = rh.get_bins(Nbins_gp,bin_limits=bin_limits)
X1,X2 = np.meshgrid(*centers_gp)

for ii, u in enumerate(shear_range_gp):
    f = model_eval.vector_field_function(driftModel)
    D = model_eval.vector_field_function(diffModel)

    p_fit = model_eval.get_stationary_probability(f,D,bins_gp,centers_gp,u,tol=p_tol)
    U= -np.log(p_fit)

    fig,ax = dynamics_viz.plot_gen_potential_2D(U,centers_gp[0],centers_gp[1],cmap='jet',surf=False)
    ax.set_xlabel(gp_args['plt_xlabel'])
    ax.set_ylabel(gp_args['plt_ylabel'])
    ax.set_title(gp_args['plt_title'])
    fig.suptitle('Shear stress: '+str(u)+' dyn/cm$^2$', y = 1.05, fontsize=16)
    vb.save_plot(fig,savedir+'figs/gp_shear_'+str(ii))

    normed = False # if True, normalize vectors by their magnitudes

    f_mesh = model_eval.mesh_grid_function(f)
    D_mesh = model_eval.mesh_grid_function(D)

    f_vals = f_mesh([X1,X2],u).T
    D_vals = D_mesh([X1,X2],u).T

    _, grad_term, _, flux_term = gp.grad_flux_decomposition(f_vals,D_vals,centers_gp,tol=p_tol)
    grad_ = grad_term.copy()
    flux_ = np.array(flux_term)
    if normed:
        grad_ = grad_/(np.sqrt(grad_[0]**2+grad_[1]**2))
        flux_ = flux_/(np.sqrt(flux_[0]**2+flux_[1]**2))

    fig,ax = dynamics_viz.plot_grad_flux_decomposition(U,centers_gp[0],centers_gp[1],
                                                       grad_,flux_,cmap='jet',normed=normed,
                                                       downsample=gp_args['downsample'])
    ax.set_xlabel(gp_args['plt_xlabel'])
    ax.set_ylabel(gp_args['plt_ylabel'])
    ax.set_title('Shear stress: '+str(u)+' dyn/cm$^2$')
    fig.suptitle(gp_args['plt_title'], y = 1.05, fontsize=16)
    vb.save_plot(fig,savedir+'figs/gp_decomp_shear_'+str(ii))

# %%
