from endo_pipeline.cli import CropPattern, Datasets, StrList
from endo_pipeline.settings.flow_field_2d import HISTOGRAM_THRESHOLD_FOR_MASKING


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    columns: StrList | None = None,
    use_same_axes: bool = False,
    mask_threshold: float | None = HISTOGRAM_THRESHOLD_FOR_MASKING,
) -> None:
    """
    Visualize DiffAE feature dynamics in 2D.

    #dynamical-systems #diffae-feature-analysis #visualization

    **Workflow defaults:**

    The defaults for the command line inputs are set to visualize drift in the
    polar radius and PC3 flipped (density proxy) for the features extracted from
    the grid-based crop pattern.

    The precomputed drift dataframes that this workflow loads by default were
    generated using the default settings for the flow field estimation workflow,
    which include using the grid-based crop pattern and computing drift in polar
    angle. If you want to visualize drift for a different variable or crop
    pattern, you must run `generate-flow-field` with the desired inputs.

    Parameters
    ----------
    crop_pattern
        The crop pattern to get features for.
    datasets
        Specific datasets to run the workflow on.
    global_limits
        Whether to use global limits for all datasets when plotting drift
        contours.
    mask_threshold
        Threshold for masking low-confidence regions of drift estimates based on
        histogram of data points. If None, no masking is applied.
    """

    import logging
    from typing import cast

    import numpy as np
    import pandas as pd

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
        slugify,
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
        mask_drift_vector_field_by_data_density,
    )
    from endo_pipeline.library.visualize.columns import get_label_for_column
    from endo_pipeline.library.visualize.diffae_features.dynamics import (
        plot_drift_contours,
        plot_drift_quiver,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        BIN_WIDTHS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        POLAR_ANGLE_PERIOD,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
        StabilityLabel,
    )
    from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, StabilityLegendHandle
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    # get labels for provided set of feature columns
    column_names_ = columns or [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_names = cast(list[Column.DiffAEData], column_names_)
    ndim = len(column_names)
    if ndim != 2:
        raise ValueError(
            f"Exactly 2 columns must be provided for 2D flow field visualization, but {ndim} were provided."
        )
    drift_column_names = [f"{name}_{Column.VectorField.DRIFT}" for name in column_names]
    column_labels = [get_label_for_column(col).replace("polar ", "") for col in column_names]
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    # get dataframe manifest for crop-based features
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[crop_pattern]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    columns_str = join_sorted_strings(cast(list[str], column_names))
    drift_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD}_{columns_str}_{base_name}"
    )
    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{columns_str}_{base_name}"
    )
    # Flexible DEMO_MODE loading pattern: first try to load the manifests with
    # the expected names, but if any of them are not found, then try to load the
    # corresponding demo manifests with the "_demo." This allows for both
    # running the full pipeline in DEMO_MODE with the demo manifests, and also
    # for running this workflow in DEMO_MODE with the full manifests if the user
    # has them available (i.e., just "demo" the visualization step without
    # needing to also "demo" the flow field estimation step).
    try:
        # Default is to load the "production" manifests, even in DEMO_MODE, to
        # allow for just "demoing" the visualization step if the full manifests
        # are available.
        drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)
        fixed_points_dataframe_manifest = load_dataframe_manifest(
            fixed_points_dataframe_manifest_name
        )
    except FileNotFoundError:
        # If the production manifests are not found, then in DEMO_MODE will try
        # to load the demo manifests with the "_demo" suffix. Else, if not in
        # DEMO_MODE, will raise the original FileNotFoundError.
        logger.warning(
            "Dataframe manifest(s) not found for production run. If you are running in DEMO_MODE, "
            "the workflow will attempt to load the corresponding demo dataframe manifest(s)."
        )
        if DEMO_MODE:
            demo_suffix = "_demo"
            drift_dataframe_manifest = load_dataframe_manifest(
                f"{drift_dataframe_manifest_name}{demo_suffix}"
            )
            fixed_points_dataframe_manifest = load_dataframe_manifest(
                f"{fixed_points_dataframe_manifest_name}{demo_suffix}"
            )
        else:
            raise

    # Use provided datasets or default if none provided.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning(
            "DEMO MODE: Processing no more than two of the provided datasets for quick visualization."
        )
        # take min of the number of datasets provided and 2, to limit to at most
        # 2 datasets in DEMO_MODE for quick visualization (i.e., avoid error if
        # only 1 dataset is provided)
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    # Get the corresponding kernels and bin widths for each variable. For the
    # polar angle variable, also specify the period for the kernel based on the
    # rescaled theta range, to ensure that the periodicity of the polar angle is
    # taken into account in the flow field estimation.
    #
    # Also initialize the plot bounds via the global bin limits dict, which will
    # be used if use_same_axes is True, and will be updated to dataset-specific
    # bin limits if use_same_axes is False
    kernels = []
    bin_widths = []
    rescaled_theta_period = POLAR_ANGLE_PERIOD + np.pi * (1 - RESCALE_THETA)
    bounds_for_plots = []
    contour_axes_titles = []
    for column_name, column_label in zip(column_names, column_labels, strict=True):
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta_period if column_name == Column.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        bin_limits_col = BIN_LIMITS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)
        bounds_for_plots.append(bin_limits_col)
        contour_axes_titles.append(f"Drift component: d{column_label}/dt")

    # loop over datasets in collection, compute 2D drift coefficients for each
    # pairwise combination of polar coordinates, and plot contours of drift coefficients
    for dataset_name in dataset_names:
        if dataset_name not in drift_dataframe_manifest.locations:
            logger.warning(
                "No drift coefficient dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping this dataset.",
                drift_dataframe_manifest_name,
                dataset_name,
            )
            continue

        logger.info(f"Visualizing flow field for dataset [ {dataset_name} ]")
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)

        # load dataframe with feature data
        # load dataframe and perform additional filtering (remove
        # non-steady-state timepoints based on annotations), computing
        # only the columns needed for flow field estimation and analysis to save memory.
        dataset_config = load_dataset_config(dataset_name)
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        feature_data = filter_dataframe_to_steady_state(df, dataset_config)

        # load drift vector field dataframe and check that required columns are
        # present
        vector_field_dataframe_location = get_dataframe_location_for_dataset(
            drift_dataframe_manifest, dataset_name
        )
        vector_field_dataframe = load_dataframe(vector_field_dataframe_location, delay=False)
        check_required_columns_in_dataframe(
            vector_field_dataframe,
            required_columns=[
                *column_names,
                *drift_column_names,
                Column.DATASET,
                Column.SHEAR_STRESS,
            ],
        )

        # load fixed point dataframe if it exists, and check that required
        # columns are present turn fixed point dataframe into list of arrays of
        # stable fixed point coordinates for each dataset to use for plotting
        dataset_has_fixed_points = False
        try:
            fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                fixed_points_dataframe_manifest, dataset_name
            )
            fixed_points_dataframe = load_dataframe(fixed_points_dataframe_location, delay=False)
            check_required_columns_in_dataframe(
                fixed_points_dataframe,
                required_columns=[
                    *column_names,
                    Column.DATASET,
                    Column.SHEAR_STRESS,
                    Column.VectorField.STABILITY,
                ],
            )
            dataset_has_fixed_points = True
        except KeyError:
            logger.warning(
                "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                "Stable fixed points will not be overlaid on the flow field visualizations for this dataset.",
                dataset_name,
                fixed_points_dataframe_manifest.name,
            )

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = slugify(f"{dataset_name}_shear_{shear_stress}")
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

            stable_fixed_points = pd.DataFrame()
            if dataset_has_fixed_points:
                fixed_points_for_flow_condition = filter_dataframe_by_shear_stress(
                    fixed_points_dataframe, shear_stress
                )
                stable_fixed_points = filter_dataframe_by_stability(
                    fixed_points_for_flow_condition, stability_label=StabilityLabel.STABLE
                )

            # get histogram for masking low-confidence regions of drift
            # estimates, using same kernels as for drift estimation, and set
            # drift to nan in low-confidence regions
            # get bin edges from bin centers and bin widths
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

            filename_prefix = f"{dataset_name_flow}{columns_str}"
            # plot drift contours and save
            fig, _ = plot_drift_contours(
                centers_mesh,
                drift,
                variable_labels=column_labels,
                axes_limits=bounds_for_plots,
                axes_titles=contour_axes_titles,
            )
            fig.suptitle(fig_title, y=1.00)
            save_plot_to_path(fig, fig_savedir, f"{filename_prefix}_contours")

            # plot quiver plot of drift and save
            fig, ax = plot_drift_quiver(
                centers_mesh,
                drift,
                variable_labels=column_labels,
                axes_limits=bounds_for_plots,
            )
            fig.suptitle(
                f"{fig_title} \n drift in ({column_labels[0]}, {column_labels[1]})",
                y=1.00,
            )
            save_plot_to_path(fig, fig_savedir, f"{filename_prefix}_quiver")

            # TO DO: turn this into a separate function
            # add_fixed_points_to_plot() that takes in the fixed point
            # dataframe, column names, and stability label to plot, and adds the
            # fixed points to the provided axis with appropriate markers and
            # legend entries based on stability
            if not stable_fixed_points.empty:
                ax.plot(
                    stable_fixed_points[column_names[0]],
                    stable_fixed_points[column_names[1]],
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
                save_plot_to_path(fig, fig_savedir, f"{filename_prefix}_quiver_fixed_points")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
