TAGS = ["diffae_features", "visualization"]


def main(dataset_collection_name: str = "pca_reference", model_name: str = "diffae_04_10") -> None:
    """Visualize key attributes of a fit PCA model."""
    import logging

    from src.endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from src.endo_pipeline.io import get_output_path, save_plot_to_path
    from src.endo_pipeline.library.analyze.diffae_manifest import (
        fit_pca,
        get_pca_loadings,
        get_timepoints_for_plotting_pcs,
    )
    from src.endo_pipeline.library.visualize.diffae_features import feature_viz
    from src.endo_pipeline.manifests import load_dataframe_manifest

    # set up logger
    logger = logging.getLogger(__name__)

    # set up output directory for figures
    fig_savedir = get_output_path(
        "pca_viz", dataset_collection_name, model_name, include_timestamp=False
    )

    # fit PCA model to the datasets in the given dataset collection
    pca = fit_pca(dataset_collection_name=dataset_collection_name, model_name=model_name)

    # plot cumulative explained variance ratio of PCA components
    fig, _ = feature_viz.plot_explained_variance(pca.explained_variance_ratio_)
    save_plot_to_path(fig, fig_savedir, "explained_variance_ratio")

    # plot PC loadings (contribution of each latent dimension to each PC)
    # first, plot for components scaled by the square root of the explained variance
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=True, magnitude=False)
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_scaled")

    # then, plot components without scaling
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=False, magnitude=False)
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_unscaled")

    # plot the absolute values of the scaled loadings
    fig, _ = feature_viz.plot_component_loadings(get_pca_loadings(pca, scaled=True, magnitude=True))
    save_plot_to_path(fig, fig_savedir, "pc_loadings_scaled_magnitude")

    # plot the absolute values of the unscaled loadings
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=False, magnitude=True)
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_unscaled_magnitude")

    # plot scatter of PCA components
    # for the datasets used to fit PCA
    # load model manifests for the given dataset collection
    dataset_names = get_datasets_in_collection(dataset_collection_name)
    pca_ref_configs = [load_dataset_config(dataset_name) for dataset_name in dataset_names]
    restrict_no_flow = True  # restrict plot to subset of no flow timepoints

    # get timepoints to use for scatter plots
    # this can definitely be written into a wrapper function
    # maybe make a dictionary instead of a list?
    timepoints_refs = get_timepoints_for_plotting_pcs(
        pca_ref_configs, restrict_no_flow=restrict_no_flow
    )

    # scatter plot of pca reference datasets
    manifest = load_dataframe_manifest(model_name)
    fig, _ = feature_viz.plot_pc_scatter(
        dataset_names, manifest, pca, timepoints_to_use=timepoints_refs
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_ref")

    logger.info("PCA visualization complete. Figures saved to [ %s ]", fig_savedir)


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
