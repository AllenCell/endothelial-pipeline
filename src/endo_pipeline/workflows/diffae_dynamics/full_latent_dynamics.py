import fire

from cellsmap.util import manifest_io
from src.endo_pipeline.configs import (
    get_model_manifest,
    get_timelapse_model_manifests,
    load_model_config,
)
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_features import regression_helper
from src.endo_pipeline.library.analyze.diffae_manifest import manifest_pca, preprocessing
from src.endo_pipeline.library.visualize import viz_base
from src.endo_pipeline.library.visualize.diffae_features import manifest_viz


def main(dataset_names: str | list[str] | None = None, model_name: str = "diffae_04_10") -> None:
    """
    Run base visualization of Diff AE latent space
    feature dynamics for a specified list of datasets.
    """
    # get model config from model name
    model_config = load_model_config(model_name)
    if dataset_names is None:
        # filter out datasets that are not timelapse
        # and load model manifests
        model_manifest_list = get_timelapse_model_manifests(model_config)
    else:
        if isinstance(dataset_names, str):
            # if a single dataset is specified, convert to list
            dataset_names = [dataset_names]
        # load manifests for the specified datasets
        model_manifest_list = [
            get_model_manifest(dataset_name, model_config) for dataset_name in dataset_names
        ]

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_name = "full_latent_dynamics"

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    fig_savedir = get_output_path(workflow_name, "model_name", "figs", include_timestamp=False)

    pca = manifest_pca.fit_pca()

    num_bins = [40, 40, 40]
    bin_limits_pcs = [[-1, 1], [-0.8, 0.7], [-0.8, 0.7]]
    bins = regression_helper.get_bins(num_bins, bin_limits=bin_limits_pcs)[0]

    for model_manifest in model_manifest_list:
        print(f"Processing dataset: {model_manifest.dataset_name}")
        df = preprocessing.get_manifest_for_dynamics_workflows(model_manifest, pca=None)
        feat_cols = manifest_io.get_feature_cols(df)
        feats = preprocessing.df_to_array(df, feat_cols)
        fig, _ = manifest_viz.plot_latent_component_mean(feats)
        fig.suptitle(f"Dataset: {model_manifest.dataset_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir / f"{model_manifest.dataset_name}_latent_mean")

        fig, _ = manifest_viz.plot_latent_component_histogram(feats)
        fig.suptitle(f"Dataset: {model_manifest.dataset_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir / f"{model_manifest.dataset_name}_latent_histogram")

        df_proj = preprocessing.project_manifest_to_pcs(df, pca)
        feats = preprocessing.df_to_array(df_proj, feat_cols)[..., :3]  # only looking at top 3 PCs

        fig, _ = manifest_viz.plot_principal_component_histogram(feats, bins=bins)
        fig.suptitle(f"Dataset: {model_manifest.dataset_name}", y=0.95, fontsize=25)
        viz_base.save_plot(fig, fig_savedir / f"{model_manifest.dataset_name}_pc_histogram")


if __name__ == "__main__":
    fire.Fire(main)
