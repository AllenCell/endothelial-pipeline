from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import Datasets, PatchType
from endo_pipeline.settings.flow_field_2d import HISTOGRAM_THRESHOLD_FOR_MASKING


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    columns: Annotated[tuple[str, str] | None, Parameter(negative_iterable=[])] = None,
    use_same_axes: Annotated[bool, Parameter(negative="--use-auto-axes")] = False,
    mask_threshold: Annotated[
        float | None, Parameter(negative_none="no-")
    ] = HISTOGRAM_THRESHOLD_FOR_MASKING,
) -> None:
    """
    Visualize 2D drift vector field and fixed points.

    #dynamical-systems #grid-based #cell-centered #visualization

    This workflow uses the precomputed drift vector field and fixed points
    output by the `generate_flow_field` workflow, run for two column names.
    Make sure to run that workflow with the matching patch type and column
    names before visualizing.

    Visualization outputs include:

    - Contour plot of each component of the drift vector field
    - Quiver plot of drift vector field over 2D state space (overlaid with
      stable fixed points, if available)

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-2d-flow-field -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-2d-flow-field --datasets DATASET_NAME
    ```

    To run the workflow for a specific columns:

    ```bash
    uv run endopipe visualize-2d-flow-field --columns COLUMN_NAME COLUMN_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will visualize the
    flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to visualize.
    columns
        Specific columns to visualize.
    use_same_axes
        True to use global limits across all datasets, False otherwise.
    mask_threshold
        Threshold for masking low-confidence regions of drift estimates based on
        histogram of data points. If None, no masking is applied.
    """

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_stability,
        filter_dataframe_to_flow_condition_by_timepoint,
        filter_dataframe_to_steady_state,
    )
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
        get_valid_flow_field_column_names,
        mask_drift_vector_field_by_data_density,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.dynamics import (
        plot_drift_contours,
        plot_drift_quiver,
    )
    from endo_pipeline.library.visualize.fixed_points import StabilityLegendHandle
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameSuffix
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        KERNEL_PERIODS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
    )
    from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Ensure that selected columns are valid options
    default_columns = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_names = get_valid_flow_field_column_names(columns, default_columns)

    # Get label and drift column name for selected column
    column_labels = [get_label_for_column(column) for column in column_names]
    drift_column_names = [f"{column}{ColumnNameSuffix.DRIFT}" for column in column_names]
    fp_column_names = [f"{column}{ColumnNameSuffix.FIXED_POINTS}" for column in column_names]
    mesh_column_names = [f"{column}{ColumnNameSuffix.MESH_GRID}" for column in column_names]

    # Required columns for vector field and fixed point manifests
    required_vector_field_columns = [
        *mesh_column_names,
        *drift_column_names,
        Column.DATASET,
        Column.SHEAR_STRESS,
    ]
    required_fixed_point_columns = [
        *fp_column_names,
        Column.DATASET,
        Column.SHEAR_STRESS,
        Column.FIXED_POINT_STABILITY,
    ]

    # Columns to keep when loading feature dataframe
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load feature dataframe for specified patch type
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Load drift vector field and fixed points for selected column
    name_suffix = f"_{join_sorted_strings(column_names)}_{patch_type}"
    vector_field_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD}{name_suffix}"
    fixed_points_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)
    fixed_points_manifest = load_dataframe_manifest(fixed_points_manifest_name)

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    bounds_for_plots: list[tuple[float, float]] = []
    contour_axes_titles: list[str] = []
    for column_name, column_label in zip(column_names, column_labels, strict=True):
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])
        bounds_for_plots.append(BIN_LIMITS_DYNAMICS[column_name])
        contour_axes_titles.append(f"Drift component: d{column_label}/dt")

    for dataset_name in dataset_names:
        # Check if dataset available in vector field manifest
        if dataset_name not in vector_field_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. Skipping.",
                dataset_name,
                vector_field_manifest_name,
            )
            continue

        # Load dataset config
        dataset_config = load_dataset_config(dataset_name)

        # Load feature dataframe for dataset with only the required columns and
        # filter out non-steady-state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        feature_data = filter_dataframe_to_steady_state(df, dataset_config)

        # Load vector field dataframe and check required columns
        vector_field_dataframe_location = get_dataframe_location_for_dataset(
            vector_field_manifest, dataset_name
        )
        vector_field_dataframe = load_dataframe(vector_field_dataframe_location, delay=False)
        check_required_columns_in_dataframe(vector_field_dataframe, required_vector_field_columns)

        # Load fixed points dataframe and check required columns, if available
        if dataset_name not in fixed_points_manifest.locations:
            logger.warning(
                "Dataset '%s' not found in manifest '%s'. "
                "Stable fixed points will not be shown in output visualization.",
                dataset_name,
                vector_field_manifest_name,
            )
            fixed_points_dataframe = None
        else:
            fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                fixed_points_manifest, dataset_name
            )
            fixed_points_dataframe = load_dataframe(fixed_points_dataframe_location, delay=False)
            check_required_columns_in_dataframe(
                fixed_points_dataframe, required_fixed_point_columns
            )

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
            fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)

            feature_data_for_flow_condition = filter_dataframe_to_flow_condition_by_timepoint(
                feature_data, dataset_config, flow_condition
            )
            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                vector_field_dataframe, shear_stress
            )

            drift, centers = get_reshaped_vector_field_and_grid(
                vector_field_for_flow_condition,
                column_names=column_names,
            )
            centers_mesh = np.meshgrid(*centers, indexing="ij")

            # get histogram for masking low-confidence regions of drift
            # estimates, using same kernels as for drift estimation, and set
            # drift to nan in low-confidence regions get bin edges from bin
            # centers and bin widths
            bin_limits = [
                (centers[i].min() - bin_widths[i] / 2, centers[i].max() + bin_widths[i] / 2)
                for i in range(len(column_names))
            ]
            if not use_same_axes:
                bounds_for_plots = bin_limits.copy()

            if mask_threshold is not None:
                bins_2d = get_bins(bin_widths=bin_widths, bin_limits=bin_limits, pad=0)[0]
                drift = mask_drift_vector_field_by_data_density(
                    drift_coeffs=drift,
                    dataframe=feature_data_for_flow_condition,
                    column_names=column_names,
                    histogram_bins=bins_2d,
                    histogram_kernel=kernels,
                    probability_threshold=mask_threshold,
                )

            # Plot contours plot of drift and save
            fig, _ = plot_drift_contours(
                centers_mesh,
                drift,
                variable_labels=column_labels,
                axes_limits=bounds_for_plots,
                axes_titles=contour_axes_titles,
            )
            fig.suptitle(fig_title, y=1.00)
            figure_name = f"{dataset_name_flow}{name_suffix}_contours"
            save_plot_to_path(fig, output_path, figure_name, tight_layout=False, file_format=".png")

            # Plot quiver plot of drift and save
            fig, ax = plot_drift_quiver(
                centers_mesh,
                drift,
                variable_labels=column_labels,
                axes_limits=bounds_for_plots,
            )
            fig.suptitle(f"{fig_title} \n drift in ({column_labels[0]}, {column_labels[1]})", y=1.0)
            figure_name = f"{dataset_name_flow}{name_suffix}_quiver"
            save_plot_to_path(fig, output_path, figure_name, file_format=".png")

            if fixed_points_dataframe is not None:
                # TODO: turn this into a separate function
                # add_fixed_points_to_plot() that takes in the fixed point
                # dataframe, column names, and stability label to plot, and adds
                # the fixed points to the provided axis with appropriate markers
                # and legend entries based on stability
                fixed_points_for_flow_condition = filter_dataframe_by_shear_stress(
                    fixed_points_dataframe, shear_stress
                )
                stable_fixed_points = filter_dataframe_by_stability(
                    fixed_points_for_flow_condition, stability_label=StabilityLabel.STABLE
                )

                ax.plot(
                    stable_fixed_points[fp_column_names[0]],
                    stable_fixed_points[fp_column_names[1]],
                    FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
                    color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    markersize=5,
                )
                # add legend entry for stable fixed points
                stable_fixed_point_handle = StabilityLegendHandle(
                    stability_label=StabilityLabel.STABLE
                )
                # add handle to existing legend if it exists, else create new
                # legend with just the stable fixed point handle
                existing_legend = ax.get_legend()
                if existing_legend is not None:
                    existing_handles, existing_labels = existing_legend.legend_handles, [
                        text.get_text() for text in existing_legend.get_texts()
                    ]
                    ax.legend(
                        handles=[*existing_handles, stable_fixed_point_handle],
                        labels=[*existing_labels, stable_fixed_point_handle.get_label()],
                        loc="upper right",
                    )
                else:
                    ax.legend(
                        handles=[stable_fixed_point_handle],
                        labels=[stable_fixed_point_handle.get_label()],
                        loc="upper right",
                    )
                figure_name = f"{dataset_name_flow}{name_suffix}_quiver_fixed_points"
                save_plot_to_path(fig, output_path, figure_name, file_format=".png")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
