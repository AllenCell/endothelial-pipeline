# %%
import numdifftools as nd
import numpy as np
import pysindy as ps

from cellsmap.analyses.utils import model_eval
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.viz import manifest_viz, pplane
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_io
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path

# %%
# get output subdirectory for intermediate workflow outputs (set in config file dynamics_config.yaml)
# if directory does not exist, get_output_path function will create it
workflow_name = "full_latent_dynamics"
workflow_output_folder = f"stochastic_dynamics/{workflow_name}/outputs"
savedir = get_output_path(workflow_output_folder)

# get output subdirectory for figures that workflow outputs (set in config file dynamics_config.yaml)
# if directory does not exist, get_output_path function will create it
workflow_fig_folder = f"stochastic_dynamics/{workflow_name}/figs"
fig_savedir = get_output_path(workflow_fig_folder)

pca = manifest_pca.fit_pca()

list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

Nbins = [40, 40, 40]
bin_limits_pcs = [[-1, 1], [-0.7, 0.7], [-1, 1]]
bins = rh.get_bins(Nbins, bin_limits=bin_limits_pcs)[0]

# %%
for ds_name in list_of_datasets:
    print(f"Processing dataset: {ds_name}")
    df_ds = diffae_preproc.get_manifest_for_dynamics_workflows(ds_name, pca=None)
    feat_cols = manifest_io.get_feature_cols(df_ds)
    feats = diffae_preproc.df_to_array(df_ds, feat_cols)
    fig, ax = manifest_viz.plot_latent_component_mean(feats)
    fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
    vb.save_plot(fig, f"{fig_savedir}/{ds_name}_latent_mean")

    fig, ax = manifest_viz.plot_latent_component_histogram(feats)
    fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
    vb.save_plot(fig, f"{fig_savedir}/{ds_name}_latent_histogram")

    df_proj = diffae_preproc.project_manifest_to_pcs(df_ds, pca)
    feats = diffae_preproc.df_to_array(df_proj, feat_cols)[
        ..., :3
    ]  # only looking at top 3 PCs

    fig, ax = manifest_viz.plot_principal_component_histogram(feats, bins=bins)
    fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
    vb.save_plot(fig, f"{fig_savedir}/{ds_name}_PC_histogram")
# %%

ds_to_analyze = "20250319_20X"
df_ = diffae_preproc.get_manifest_for_dynamics_workflows(ds_to_analyze, pca=None)
df_.sort_values(by=["crop_index", "frame_number"], inplace=True)
latent_idxs = [3, 4, 6]
latent_dims = [f"feat_{i}" for i in latent_idxs]

X_list, dX_list, dT_list = rh.get_X_dX_and_dT(df_, latent_dims)

# %%
Nbins = [50 for _ in range(len(latent_dims))]
bins, centers = rh.get_bins(Nbins, data=X_list)
kernel_params = {"bandwidth": 0.05, "kernel": "gaussian"}

# get drift and diffusion estimates (Kramers-Moyal coefficients)
f_KM_, D_KM_ = rh.get_kramers_moyal(
    X_list, dX_list, dT_list, bins, dt=5, kernel_params=kernel_params
)

# plot drift and diffusion estimates
ndim = len(latent_dims)
if ndim == 2:
    kmc = np.concatenate([f_KM_, D_KM_], axis=-1).T
    fig, ax00, ax01, ax10, ax11 = manifest_viz.plot_km(centers, kmc, latent_idxs, 12.2)
    ax = (ax00, ax01, ax10, ax11)
    for ax_ in ax:
        ax_.set_xlabel(f"latent dim {latent_idxs[0]}", fontsize=15)
        ax_.set_ylabel(f"latent dim {latent_idxs[1]}", fontsize=15)

    vb.save_plot(
        fig, filename=fig_savedir + f"kmcs_all_{ds_name}", format=".png", dpi=500
    )

    # quiver and streamplot of drift vector field
    fig, ax = manifest_viz.plot_km_drift_2D(centers, kmc, latent_idxs, 12.2)
    for ax_ in ax:
        ax_.set_xlabel(f"latent dim {latent_idxs[0]}", fontsize=15)
        ax_.set_ylabel(f"latent dim {latent_idxs[1]}", fontsize=15)
    vb.save_plot(
        fig, filename=fig_savedir + f"kmcs_drift_{ds_name}", format=".png", dpi=500
    )

# %%
(
    f_KM_noNAN,
    X_pts_,
) = rh.masked_vector_field(f_KM_, np.array(np.meshgrid(*centers)).T)
D_KM_noNAN, _ = rh.masked_vector_field(D_KM_, np.array(np.meshgrid(*centers)).T)

del f_KM_, D_KM_

X_train, X_test, Y_train, Y_test, V_train, V_test = rh.train_test_all(
    [X_pts_], [f_KM_noNAN], [D_KM_noNAN]
)

del f_KM_noNAN, D_KM_noNAN, X_pts_
# %%
# for fitting model of drift and diffusion terms
drift_lib = ps.PolynomialLibrary(degree=5, include_bias=True)

diff_lib = ps.PolynomialLibrary(degree=5, include_bias=True)
################### Fit SINDy models ###################

# fit model for drift term - SINDy based regression
driftModel = ps.SINDy(feature_library=drift_lib, optimizer=ps.SSR())
driftModel.fit(X_train, t=5, x_dot=Y_train)

# score on test set
drift_R2 = driftModel.score(X_test, x_dot=Y_test)
driftModel.print()

print("Coefficient of determination (R^2) for model of drift term: %f" % drift_R2)

# fit model for diffusion term - SINDy based regression
diffModel = ps.SINDy(feature_library=diff_lib, optimizer=ps.SSR())
diffModel.fit(X_train, t=5, x_dot=V_train)

# score on test set
diff_R2 = diffModel.score(X_test, x_dot=V_test)
diffModel.print()

print("Coefficient of determination (R^2) for model of diffusion term: %f" % diff_R2)

del X_train, X_test, Y_train, Y_test, V_train, V_test
# %%
f = model_eval.vector_field_function(driftModel)
D = model_eval.vector_field_function(diffModel)

myModel = [f, D]

# save model to output directory
model_dict = {"driftModel": driftModel, "diffModel": diffModel}
dynamics_io.save_model(model_dict, savedir)
# %%
# get initial conditions by sub sampling points from centers
centers_coarse = [centers[i][::10] for i in range(len(centers))]
init_coarse = np.array(np.meshgrid(*centers_coarse)).T.reshape(-1, len(latent_dims))

# %%
fpts = pplane.get_fps(f, init_coarse)  # get fixed points
# %%
fpts_new = []
fpt_types = []
flowJacobian = nd.Jacobian(f)

# define Jacobian as a function of x - for getting stability:
for fpt in fpts:
    J = flowJacobian(fpt)
    eigvals = np.linalg.eigvals(J)
    print(f"Eigenvalues of Jacobian at fixed point {fpt}: {eigvals}")
    # check for stability of fixed points
    if np.all(np.real(eigvals) < 0):
        fptStability = "Stable"
        print(f"  • {fptStability} at x = {[fpt_i for fpt_i in fpt]}")
        fpts_new.append(fpt)
# %%
