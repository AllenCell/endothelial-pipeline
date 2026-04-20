from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid", datasets: Datasets | None = None, column: str | None = None
) -> None:
    """
    Workflow to compute and visualize 1D drift in a given variable.

    **Workflow defaults:**

    The defaults for the command line inputs are set to visualize drift in polar
    angle for the features extracted from the grid-based crop pattern for the
    dataset as set by `DEFAULT_DATASET_DYNAMICS_VIS`.

    The defaults for the model manifest name and run name are not exposed as
    command line inputs for this workflow, and are set to
    `DEFAULT_MODEL_MANIFEST_NAME` and `DEFAULT_MODEL_RUN_NAME`, respectively.

    The default bin widths, limits, and kernel bandwidths for computing the
    drift are set in the settings for dynamics workflows, and are determined
    based on the column name (see `endo_pipeline.settings.dynamics_workflows`).

    For non-polar angle columns, the limits are determined based on the data and
    the `BIN_LIMIT_PERCENTILE_CUTOFF` value, which sets the lower and upper
    percentiles to use for determining the limits.

    Parameters
    ----------
    crop_pattern
        The crop pattern for the features to visualize.
    datasets
        The dataset(s) to visualize.
    column
        The column name for the variable to compute drift for.
    """

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_flow_condition
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import plot_drift_1d
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import DEFAULT_DATASETS_DYNAMICS_VIS
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_DRIFT,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLOR_DICT,
        STABILITY_MARKER_DICT,
        StabilityLabel,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # get label for provided feature column
    column_name = column or Column.DiffAEData.POLAR_ANGLE
    column_label = get_label_for_column(column_name).replace("polar ", "")
    drift_column_name = f"{column_name}_{Column.VectorField.DRIFT}"

    # get dataframe manifest for precomputed drift and fixed points dataframes
    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{column_name}_{base_name}"
    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{column_name}_{base_name}"
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

    # Use provided datasets or default if none provided.
    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    # loop over datasets in collection, compute 1D drift for given variable, and
    # plot results, skipping datasets not found in manifest
    for dataset_name in dataset_names:
        if dataset_name not in drift_dataframe_manifest.locations:
            logger.warning(
                f"Dataset {dataset_name} not found in manifest {drift_dataframe_manifest.name}. Skipping."
            )
            continue
        fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        drift_dataframe_location = get_dataframe_location_for_dataset(
            drift_dataframe_manifest, dataset_name
        )
        drift_dataframe = load_dataframe(drift_dataframe_location, delay=False)
        check_required_columns_in_dataframe(
            drift_dataframe,
            required_columns=[
                column_name,
                drift_column_name,
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
                    column_name,
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

        # compute on a per-shear stress condition basis
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            drift_dataframe_flow = drift_dataframe[
                drift_dataframe[Column.SHEAR_STRESS] == shear_stress
            ]

            drift, centers = get_reshaped_vector_field_and_grid(
                drift_dataframe_flow,
                column_names=[column_name],
            )

            fig, ax = plot_drift_1d(
                centers=centers[-1],
                drift=drift,
                axes_labels=[column_label, f"d{column_label}/dt"],
                drift_line_kwargs={"color": "k", "linewidth": 2},
                zero_line_kwargs={"linestyle": "--", "color": "gray", "linewidth": 1, "alpha": 0.7},
            )
            ax.set_title(fig_title)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_drift_{column_name}.png")

            if dataset_has_fixed_points:
                fixed_points_dataframe_flow = filter_dataframe_by_flow_condition(
                    fixed_points_dataframe, dataset_name, shear_stress
                )
                stable_fixed_points = fixed_points_dataframe_flow[
                    fixed_points_dataframe_flow[Column.VectorField.STABILITY]
                    == StabilityLabel.STABLE
                ]
                ax.plot(
                    stable_fixed_points[column_name],
                    np.zeros_like(stable_fixed_points[column_name]),
                    STABILITY_MARKER_DICT[StabilityLabel.STABLE],
                    color=STABILITY_COLOR_DICT[StabilityLabel.STABLE],
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    markersize=5,
                )
                save_plot_to_path(
                    fig,
                    fig_savedir,
                    f"{dataset_name_flow}_drift_{column_name}_stable_fixed_points.png",
                )

        if DEMO_MODE:
            logger.warning("DEMO MODE: only running workflow on first available dataset.")
            break


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
