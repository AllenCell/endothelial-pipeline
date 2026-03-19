from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.migration_coherence import DEFAULT_MIGRATION_COHERENCE_FEATURE


def main(
    datasets: Datasets | None = None,
    crop_pattern: CropPattern = "grid",
    optical_flow_feature: str = DEFAULT_MIGRATION_COHERENCE_FEATURE,
) -> None:
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
        list_datasets_with_dataframes,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.migration_coherence import (
        MINIMUM_TRACK_LENGTH_FOR_MIGRATION_COHERENCE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # Load diffae features
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    pca = fit_pca(num_pcs=3)

    # Default list of datasets if not provided, only include datasets available
    # in the provided dataframe manifest
    valid_dataset_options = list_datasets_with_dataframes(dataframe_manifest)
    if datasets is None:
        # these collections are mutually exclusive, so we don't have to worry
        # about duplicates when concatenating
        dataset_names = get_datasets_in_collection(
            "diffae_model_training", valid_dataset_options
        ) + get_datasets_in_collection("replicate_2_datasets", valid_dataset_options)
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    # if in demo mode, only process the first dataset and log a warning
    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning(
            "Running in demo mode, only processing first dataset [ %s ]",
            dataset_names[0],
        )

    # If using tracked crops, pass in minimum track length to filter for tracks
    # of sufficient length for migration coherence analyses. If using grid
    # crops, this parameter will be ignored.
    minimum_track_length = (
        MINIMUM_TRACK_LENGTH_FOR_MIGRATION_COHERENCE if crop_pattern == "tracked" else None
    )

    # Load optical flow features and plot against diffae features
    for dataset_name in dataset_names:
        output_dir = get_output_path(__file__, dataset_name)

        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
            minimum_track_length=minimum_track_length,
        )
        df_of = add_optical_flow_features(
            df_dataset,
            datasets=[dataset_name],
        )
        plot_optical_flow_feature_distribution(
            df=df_of,
            optical_flow_feature=optical_flow_feature,
            datasets=[dataset_name],
            output_dir=output_dir,
            binwidth=0.02,
            bins=50,
            kde=True,
        )
        for x_col, y_col in [
            (ColumnName.POLAR_RADIUS, ColumnName.POLAR_ANGLE),
            (ColumnName.PC3_FLIPPED, ColumnName.POLAR_ANGLE),
            (ColumnName.POLAR_RADIUS, ColumnName.PC3_FLIPPED),
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
                color_col=optical_flow_feature,
                output_dir=output_dir,
                vmax=1,
                vmin=0,
                x_bin_size=0.25,
                y_bin_size=0.25,
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
