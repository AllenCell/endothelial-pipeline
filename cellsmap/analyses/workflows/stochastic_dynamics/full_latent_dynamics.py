# %%
from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from cellsmap.analyses.utils import regression_main
from cellsmap.analyses.utils.io import dynamics_io
from cellsmap.analyses.utils.viz import manifest_viz, viz_base as vb

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

# load manifest to DataFrame with metadata
df = manifest_io.load_manifest_to_df()

# %%
feat_cols = [str(i) for i in range(8)]

list_of_datasets = manifest_io.get_list_of_datasets(df,verbose=False)

for ds_name in list_of_datasets:
    print(f"Processing dataset: {ds_name}")
    df_ds = df[df['dataset_name'] == ds_name].copy()
    df_ds = manifest_io.add_crop_index(df_ds)
    feats = manifest_io.df_to_array(df_ds, feat_cols)
    fig, ax = manifest_viz.plot_latent_component_mean(feats)
    fig.suptitle(f"Dataset: {ds_name}")
    vb.save_plot(fig,f"{fig_savedir}/{ds_name}_latent_mean")

    fig, ax = manifest_viz.plot_latent_component_histogram(feats)
    fig.suptitle(f"Dataset: {ds_name}")
    vb.save_plot(fig,f"{fig_savedir}/{ds_name}_latent_histogram")
# %%
