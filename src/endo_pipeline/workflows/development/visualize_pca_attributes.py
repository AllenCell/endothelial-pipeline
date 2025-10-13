from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.settings import DEFAULT_MODEL_MANIFEST_NAME, DEFAULT_MODEL_RUN_NAME

TAGS = ["diffae_features", "visualization"]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    include_cell_piling: Annotated[bool, Parameter(negative="--exclude-cell-piling")] = False,
    dataset_collection_name: str = "pca_reference",
) -> None:
    """Visualize key attributes of a fit PCA model."""
    import logging

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe import (
        fit_pca,
        get_pca_loadings,
        get_pca_loadings_as_df,
    )
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        get_most_recent_run_name,
        load_dataframe_manifest,
        load_model_manifest,
    )

    # set up logger
    logger = logging.getLogger(__name__)

    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )

    # set up output directory for figures
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    fig_savedir = get_output_path(
        "pca_viz", dataset_collection_name, model_manifest_name, run_name_
    )

    # fit PCA model to the datasets in the given dataset collection
    logger.debug(
        "Fitting PCA model to datasets in collection [ %s ] "
        "using features from dataframe manifest [ %s ]",
        dataset_collection_name,
        dataframe_manifest_name,
    )
    pca = fit_pca(
        dataset_collection_name=dataset_collection_name,
        dataframe_manifest_name=dataframe_manifest_name,
        include_cell_piling=include_cell_piling,
    )

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

    # scatter plot of pca reference datasets
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    fig, _ = feature_viz.plot_pc_scatter(
        dataset_names,
        dataframe_manifest,
        pca,
        include_cell_piling=include_cell_piling,
    )
    save_plot_to_path(fig, fig_savedir, "pca_scatter_ref")

    # heatmap and clustemap of PC loadings
    pca_loadings_df = get_pca_loadings_as_df(pca, df_format="wide")
    fig_heatmap = feature_viz.pc_loading_heatmap_workflow(pca_loadings_df)
    save_plot_to_path(
        figure=fig_heatmap,
        output_path=fig_savedir,
        figure_name="pca_loadings_heatmap",
    )

    logger.info("PCA visualization complete. Figures saved to [ %s ]", fig_savedir)


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
