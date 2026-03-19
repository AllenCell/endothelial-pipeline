def main():
    import logging

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.analyze.migration_pc.optical_flow_feature import (
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.migration_coherence import (
        plot_optical_flow_feature_distribution,
        plot_scatter_and_binned_heatmap,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    OPTICAL_FLOW_FEATURE = "optical_flow_mean_unit_vector_dt1"

    OPTICAL_FLOW_MANIFEST_NAME = "optical_flow_bf"

    CROP_PATTERN = "grid"

    datasets = get_datasets_in_collection("diffae_model_training") + get_datasets_in_collection(
        "replicate_2_datasets"
    )
    if DEMO_MODE:
        datasets = datasets[:1]

    output_dir = get_output_path("migration_coherence")

    # Load diffae features
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern=CROP_PATTERN
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    pca = fit_pca(num_pcs=3)

    # Load optical flow features and plot against diffae features
    for dataset_name in datasets:
        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=CROP_PATTERN,
        )
        df_of = add_optical_flow_features(
            df_dataset,
            datasets=[dataset_name],
            optical_flow_manifest_name=OPTICAL_FLOW_MANIFEST_NAME,
        )
        plot_optical_flow_feature_distribution(
            df=df_of,
            optical_flow_feature=OPTICAL_FLOW_FEATURE,
            datasets=[dataset_name],
            output_dir=output_dir,
            binwidth=0.02,
            bins=50,
            kde=True,
        )
        for x_col, y_col in [
            (DYNAMICS_COLUMN_NAMES[0], DYNAMICS_COLUMN_NAMES[1]),
            (DYNAMICS_COLUMN_NAMES[0], DYNAMICS_COLUMN_NAMES[2]),
            (DYNAMICS_COLUMN_NAMES[1], DYNAMICS_COLUMN_NAMES[2]),
        ]:
            logger.info(
                "Plotting optical flow feature over [ %s ] vs [ %s ] for dataset [ %s ]",
                x_col,
                y_col,
                dataset_name,
            )
            plot_scatter_and_binned_heatmap(
                df=df_of,
                dataset_name=dataset_name,
                x_col=x_col,
                y_col=y_col,
                color_col=OPTICAL_FLOW_FEATURE,
                output_dir=output_dir,
                vmax=1,
                vmin=0,
                x_bin_size=0.25,
                y_bin_size=0.25,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
