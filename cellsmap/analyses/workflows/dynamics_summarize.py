# %%
from cellsmap.analyses.utils import model_analysis, regression_helper as rh
from cellsmap.analyses.utils.io import dynamics_io

from cellsmap.analyses.configs.manifest_postproc_config import savedir, ds_to_skip, PCs
from cellsmap.analyses.configs.dynamics_viz_config import Nbins_plot, bin_limits, \
    pplane_args, shear_range, fpt_args, Nbins_gp, shear_range_gp, gp_args

# %%
# load fitted drift-diffusion model (list of fit SINDy objects)
model_dict = dynamics_io.load_model(savedir+'outputs/drift_diffusion_model.pkl')

driftModel = model_dict['driftModel']
diffModel = model_dict['diffModel']
myModel = [driftModel,diffModel]
# %%
# for plotting phase plane and histogram plots, fix grid and bin limits across all datasets
bins, centers = rh.get_bins(Nbins_plot,bin_limits=bin_limits)

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

# get bins and centers for plotting generalized potential energy landscape (fixed across all values of shear stress)
bins_gp, centers_gp = rh.get_bins(Nbins_gp,bin_limits=bin_limits)

# plot generalized potential energy landscape for each shear stress specified in shear_range_gp
model_analysis.run_gen_potential_analysis(myModel,bins_gp,centers_gp,shear_range_gp,gp_args,savedir)

# %%
