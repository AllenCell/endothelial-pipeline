def main(
    num_pcs: int | None = None,
    num_features: int | None = None,
    include_loadings_legend: bool = False,
    annotate: bool = True,
) -> None:
    """
    Visualize key attributes of a fit PCA model.

    #diffae_features #visualization #pca

    **Workflow defaults**
    - model_manifest_name: DEFAULT_MODEL_MANIFEST_NAME
    - run_name: DEFAULT_MODEL_RUN_NAME
    - datasets: DEFAULT_PCA_DATASET_COLLECTION_NAME
    - crop_pattern: "grid"
    - num_pcs: NUM_PCS_TO_ANALYZE
    - num_features: NUM_LATENT_FEATURES

    Parameters
    ----------
    num_pcs
        Number of principal components to analyze and visualize.
    num_features
        Number of latent features to include in the PCA loading visualizations.
    include_loadings_legend
        If true, include a legend in the PCA loadings plots.
    annotate
        If true, annotate the heatmap and clustermap of PCA loadings.
    """

    from endo_pipeline.configs import get_latent_dim_from_config
    from endo_pipeline.io import get_config_dict_from_mlflow, get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_pca_loadings,
        get_pca_loadings_as_df,
    )
    from endo_pipeline.library.visualize.diffae_features import feature_viz
    from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
        plot_and_save_clustermap,
    )
    from endo_pipeline.manifests import get_model_location_for_run, load_model_manifest
    from endo_pipeline.settings.diffae_feature_dataframes import (
        DIFFAE_FEATURE_COLUMN_NAMES,
        DIFFAE_PC_COLUMN_NAMES,
        NUM_LATENT_FEATURES,
        NUM_PCS_TO_ANALYZE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    # load model manifest for default model manifest name and run name
    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    model_manifest = load_model_manifest(model_manifest_name)

    # get latent dimension from model config
    model_location = get_model_location_for_run(model_manifest, run_name)
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    num_latent_dim = num_features or min(
        NUM_LATENT_FEATURES, get_latent_dim_from_config(model_config)
    )
    num_pc_dim = num_pcs or min(NUM_PCS_TO_ANALYZE, num_latent_dim)
    feat_col_names = DIFFAE_FEATURE_COLUMN_NAMES[:num_latent_dim]
    pc_col_names = DIFFAE_PC_COLUMN_NAMES[:num_pc_dim]

    # set up output directory for figures
    fig_savedir = get_output_path(__file__)

    # fit PCA model with the specified number of components using the method
    # defaults
    pca = fit_pca(num_pcs=num_pc_dim)

    # plot cumulative explained variance ratio of PCA components
    fig, _ = feature_viz.plot_explained_variance(pca.explained_variance_ratio_)
    save_plot_to_path(fig, fig_savedir, "explained_variance_ratio")

    # plot PC loadings (contribution of each latent dimension to each PC)
    # first, plot for components scaled by the square root of the explained variance
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=True, magnitude=False), include_legend=include_loadings_legend
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_scaled")

    # then, plot components without scaling
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=False, magnitude=False), include_legend=include_loadings_legend
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_unscaled")

    # plot the absolute values of the scaled loadings
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=True, magnitude=True), include_legend=include_loadings_legend
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_scaled_magnitude")

    # plot the absolute values of the unscaled loadings
    fig, _ = feature_viz.plot_component_loadings(
        get_pca_loadings(pca, scaled=False, magnitude=True), include_legend=include_loadings_legend
    )
    save_plot_to_path(fig, fig_savedir, "pc_loadings_unscaled_magnitude")

    # heatmap and clustermap of PC loadings
    pca_loadings_df = get_pca_loadings_as_df(pca, df_format="wide")
    fig_heatmap = feature_viz.pc_loading_heatmap_workflow(
        pca_loadings_df,
        diffae_feature_columns=feat_col_names,
        pc_columns=pc_col_names,
        annotate=annotate,
    )
    save_plot_to_path(
        figure=fig_heatmap,
        output_path=fig_savedir,
        figure_name="pca_loadings_heatmap",
    )

    plot_and_save_clustermap(
        df=pca_loadings_df,
        output_folder=fig_savedir,
        metric="correlation",
        filename="pca_loadings_clustermap",
        data_type="samples",
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
