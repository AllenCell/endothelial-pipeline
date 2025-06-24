import fire

from cellsmap.util import manifest_io
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.visualize import viz_base as vb
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(list_of_datasets: list[str] | None = None) -> None:
    """
    Run base visualization of Diff AE latent space
    feature dynamics for a specified list of datasets.
    """
    if list_of_datasets is None:
        list_of_datasets = manifest_io.list_datasets_with_manifest("diffae_manifest_fmsid")
    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_name = "full_latent_dynamics"

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_fig_folder = f"stochastic_dynamics/{workflow_name}/figs"
    fig_savedir = get_output_path(workflow_fig_folder)

    pca = manifest_pca.fit_pca()

    num_bins = [40, 40, 40]
    bin_limits_pcs = [[-1, 1], [-0.8, 0.7], [-0.8, 0.7]]
    bins = rh.get_bins(num_bins, bin_limits=bin_limits_pcs)[0]

    for ds_name in list_of_datasets:
        print(f"Processing dataset: {ds_name}")
        df_ds = diffae_preproc.get_manifest_for_dynamics_workflows(ds_name, pca=None)
        feat_cols = manifest_io.get_feature_cols(df_ds)
        feats = diffae_preproc.df_to_array(df_ds, feat_cols)
        fig, _ = manifest_viz.plot_latent_component_mean(feats)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        vb.save_plot(fig, f"{fig_savedir}/{ds_name}_latent_mean")

        fig, _ = manifest_viz.plot_latent_component_histogram(feats)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        vb.save_plot(fig, f"{fig_savedir}/{ds_name}_latent_histogram")

        df_proj = diffae_preproc.project_manifest_to_pcs(df_ds, pca)
        feats = diffae_preproc.df_to_array(df_proj, feat_cols)[..., :3]  # only looking at top 3 PCs

        fig, _ = manifest_viz.plot_principal_component_histogram(feats, bins=bins)
        fig.suptitle(f"Dataset: {ds_name}", y=0.95, fontsize=25)
        vb.save_plot(fig, f"{fig_savedir}/{ds_name}_PC_histogram")


if __name__ == "__main__":
    fire.Fire(main)
