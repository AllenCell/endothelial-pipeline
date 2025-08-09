def main(dataset_names: str | list[str] | None = None, model_name: str = "diffae_04_10") -> None:
    """
    Run base visualization of Diff AE latent space
    feature dynamics for a specified list of datasets.
    """
    from src.endo_pipeline.configs import (
        get_model_manifest,
        get_timelapse_model_manifests,
        load_model_config,
    )
    from src.endo_pipeline.io import get_output_path, save_plot_to_path
    from src.endo_pipeline.library.analyze.diffae_manifest import (
        df_to_array,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_feature_column_names,
        get_pc_column_names,
        project_manifest_to_pcs,
    )
    from src.endo_pipeline.library.analyze.numerics import get_bins
    from src.endo_pipeline.library.visualize.diffae_features import feature_viz

    # get model config from model name
    model_config = load_model_config(model_name)
    if dataset_names is None:
        # filter out datasets that are not timelapse
        # and load model manifests
        model_manifest_list = get_timelapse_model_manifests(model_config)
    else:
        dataset_name_list = dataset_names

    # get output subdirectory for intermediate workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    workflow_name = "full_latent_dynamics"

    # get output subdirectory for figures that workflow outputs
    # (set in config file dynamics_config.yaml)
    # if directory does not exist, get_output_path function will create it
    fig_savedir = get_output_path(workflow_name, "model_name", "figs", include_timestamp=False)

    pca = fit_pca(model_name=model_name)

    num_bins = [40, 40, 40]
    bin_limits_pcs = [[-1, 1], [-0.8, 0.7], [-0.8, 0.7]]
    bins = get_bins(num_bins, bin_limits=bin_limits_pcs)[0]

    for dataset_name in dataset_name_list:
        print(f"Processing dataset: {dataset_name}")
        df = get_dataframe_for_dynamics_workflows(dataset_name, manifest, pca=None)
        feature_column_names = get_feature_column_names(df)
        feats = df_to_array(df, feature_column_names)
        fig, _ = feature_viz.plot_latent_component_mean(feats)
        fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
        save_plot_to_path(fig, fig_savedir, f"{dataset_name}_latent_mean")

        fig, _ = feature_viz.plot_latent_component_histogram(feats)
        fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
        save_plot_to_path(fig, fig_savedir, f"{dataset_name}_latent_histogram")

        df_proj = project_manifest_to_pcs(df, pca, feat_cols=feature_column_names)
        pc_column_names = get_pc_column_names(df_proj, pc_axes=[0, 1, 2])
        feats = df_to_array(df_proj, pc_column_names)  # only looking at top 3 PCs

        fig, _ = feature_viz.plot_principal_component_histogram(feats, bins=bins)
        fig.suptitle(f"Dataset: {dataset_name}", y=0.95, fontsize=25)
        save_plot_to_path(fig, fig_savedir, f"{dataset_name}_pc_histogram")


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
