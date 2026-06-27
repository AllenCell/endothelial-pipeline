from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Plot segmentation features for given datasets.

    #cdh5-segmentation #cdh5-tracking #nuclei-prediction #test-ready

    The following features are plotted:

    - alignment
    - orientation
    - cell migration angle
    - nuclei orientation relative to flow
    - nuclei orientation relative to migration
    - nematic order
    - eccentricity
    - aspect ratio
    - cell area
    - number of neighbors
    - centroid velocity
    - nuclei-cell centroid distance
    - number of nuclei at each timepoint
    - number of tracks after filtering

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-segmentation-features -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe plot-segmentation-features --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `live_cdh5_seg_based_feat_datasets` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will plot features
    for a single position of a single dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to plot.
    """

    import logging
    from collections import namedtuple

    import matplotlib.pyplot as plt

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        calculate_derived_data_dynamics_dependent,
    )
    from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
        mark_parallel,
        mark_perpendicular,
        plot_histogram_of_features,
        plot_line_of_features,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_SEG_FEATURE_MANIFEST_NAME,
        SEGMENTATION_FEATURE_COLUMNS,
    )

    logger = logging.getLogger(__name__)
    plt.style.use("endo_pipeline.figure")

    # Set up list of features to plot and corresponding flags.
    PlotFlags = namedtuple("PlotFlags", ["make_line", "make_hist", "is_angular"])
    plot_both_angular = PlotFlags(make_line=True, make_hist=True, is_angular=True)
    plot_both = PlotFlags(make_line=True, make_hist=True, is_angular=False)
    plot_line_only = PlotFlags(make_line=True, make_hist=False, is_angular=False)
    features_to_plot = {
        Column.SegData.ALIGNMENT_DEG: plot_both_angular,
        Column.SegData.ORIENTATION_DEG: plot_both_angular,
        Column.SegData.CENTROID_VELOCITY_ANGLE_DEG: plot_both_angular,
        Column.SegData.NUCLEI_POSITION_ANGLE_DEG: plot_both_angular,
        Column.SegData.NUCLEI_POSITION_RELATIVE_MIGRATION_DEG: plot_both_angular,
        Column.SegData.NEMATIC_ORDER: plot_both,
        Column.SegData.ECCENTRICITY: plot_both,
        Column.SegData.ASPECT_RATIO: plot_both,
        Column.SegData.AREA_UM_SQ: plot_both,
        Column.SegData.NUM_NEIGHBORS: plot_both,
        Column.SegData.CENTROID_VELOCITY_UM_PER_MIN: plot_both,
        Column.SegData.NUCLEI_POSITION_DISTANCE: plot_both,
        Column.SegData.NUM_NUCLEI_AT_TIMEPOINT: plot_line_only,
        Column.SegData.NUM_TRACKS_AFTER_FILTERING: plot_line_only,
    }

    dataset_names = datasets or get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    # If running in demo mode, only process the first dataset.
    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]
        max_positions = 1
    else:
        max_positions = None

    # Create output directories
    output_path_line = get_output_path(__file__, "lineplots")
    output_path_hist = get_output_path(__file__, "histplots")

    # Get list of columns to load including: columns required for calculating
    # dynamics features, features to be plotted, and the IS_INCLUDED filter
    all_column_names = set(
        SEGMENTATION_FEATURE_COLUMNS["dynamics_calculation_prereq"]
        + list(features_to_plot.keys())
        + [Column.SegDataFilters.IS_INCLUDED]
    )

    for dataset_name in dataset_names:
        # Load dataset config.
        dataset_config = load_dataset_config(dataset_name)

        # Load the segmentation features table
        df_manifest = load_dataframe_manifest(DEFAULT_SEG_FEATURE_MANIFEST_NAME)
        df_location = get_dataframe_location_for_dataset(df_manifest, dataset_name)
        df_delay = load_dataframe(df_location, delay=True)

        # Compute selected features from the dataframes
        cols_to_compute = list(set(all_column_names) & set(df_delay.columns))
        df = df_delay[cols_to_compute].compute().reset_index()

        # Apply global "is included" filter and then calculate derived features
        df = df[df[Column.SegDataFilters.IS_INCLUDED]]
        df = calculate_derived_data_dynamics_dependent(df)

        # Get metadata for x axis feature (time in hours)
        x_feature = Column.SegData.TIME_HRS
        x_metadata = COLUMN_METADATA[x_feature]

        positions = dataset_config.zarr_positions
        if max_positions is not None:
            positions = positions[:max_positions]

        # Iterate over each position in the dataset and generate plots
        for position in positions:
            logger.info(f"Plotting features for dataset '{dataset_name}' position '{position}'")
            df_for_position = df[df[Column.POSITION] == position]

            for y_feature, plot_flags in features_to_plot.items():
                y_metadata = COLUMN_METADATA[y_feature]
                feature_filename = f"{dataset_name}_P{position}_{y_metadata.slug}"

                if plot_flags.make_line:
                    fig_line, _ = plot_line_of_features(
                        df=df_for_position,
                        x_column_name=x_feature,
                        y_column_name=y_feature,
                        x_feature_metadata=x_metadata,
                        y_feature_metadata=y_metadata,
                        x_minor_ticks=True,
                        y_minor_ticks=True,
                    )
                    save_plot_to_path(
                        fig_line, output_path_line, feature_filename, tight_layout=False
                    )

                if plot_flags.make_hist:
                    fig_hist, ax_hist = plot_histogram_of_features(
                        df=df_for_position,
                        x_column_name=x_feature,
                        y_column_name=y_feature,
                        x_feature_metadata=x_metadata,
                        y_feature_metadata=y_metadata,
                        x_minor_ticks=True,
                        y_minor_ticks=True,
                    )

                    if plot_flags.is_angular:
                        mark_parallel(ax_hist)
                        mark_perpendicular(ax_hist)

                    save_plot_to_path(
                        fig_hist, output_path_hist, feature_filename, tight_layout=False
                    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
