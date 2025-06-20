# %%
from cellsmap.analyses.utils import regression_helper as rh
from cellsmap.analyses.utils.viz import manifest_viz
from cellsmap.analyses.utils.viz import viz_base as vb
from cellsmap.util import manifest_io
from cellsmap.util.manifest_preprocessing import (
    diffae_feature_preprocessing as diffae_preproc,
)
from cellsmap.util.manifest_preprocessing import manifest_pca
from cellsmap.util.set_output import get_output_path

# %%
# get output subdirectory for intermediate workflow outputs
# (set in config file dynamics_config.yaml)
# if directory does not exist, get_output_path function will create it
workflow_name = "full_latent_dynamics"
workflow_output_folder = f"stochastic_dynamics/{workflow_name}/outputs"
savedir = get_output_path(workflow_output_folder)

# get output subdirectory for figures that workflow outputs
# (set in config file dynamics_config.yaml)
# if directory does not exist, get_output_path function will create it
workflow_fig_folder = f"stochastic_dynamics/{workflow_name}/figs"
fig_savedir = get_output_path(workflow_fig_folder)

pca = manifest_pca.fit_pca()

list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")

Nbins = [40, 40, 40]
bin_limits_pcs = [[-1, 1], [-0.8, 0.7], [-0.8, 0.7]]
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
