from endo_pipeline.cli import Datasets, PatchType


def main(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    column: str | None = None,
) -> None:
    """
    Visualize 1D drift vector field and fixed points.

    #dynamical-systems #grid-based #cell-centered #visualization #test-ready

    This workflow uses the precomputed drift vector field and fixed points
    output by the `generate_flow_field` workflow, run for a single column name.
    Make sure to run that workflow with the matching patch type and column
    name before visualizing.

    Visualization outputs include:

    - Line plot of drift vector field across 1D state space (overlaid with
      stable fixed points, if available)

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe visualize-1d-flow-field -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe visualize-1d-flow-field --datasets DATASET_NAME
    ```

    To run the workflow for a specific column:

    ```bash
    uv run endopipe visualize-1d-flow-field --column COLUMN_NAME
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
        Specific column to visualize.
    """

    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        get_datasets_in_collection,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_shear_stress,
        filter_dataframe_by_stability,
    )
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_reshaped_vector_field_and_grid,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics import (
        make_legend_handles_for_fixed_pts,
        plot_drift_1d,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
    )
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD,
    )
    from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    # Ensure that selected column is valid option.
    if column is None:
        column_name = Column.DiffAEData.POLAR_ANGLE
    elif column not in DYNAMICS_COLUMN_NAMES:
        logger.error("Column '%s' not supported for flow field visualization. Exiting.", column)
        return
    else:
        column_name = Column.DiffAEData(column)

    # Get label and drift column name for selected column
    column_label = get_label_for_column(column_name)
    drift_column_name = ColumnTemplate.DRIFT_COEFFICIENT % column_name
    fp_column_name = ColumnTemplate.FIXED_POINT % column_name
    mesh_column_name = ColumnTemplate.MESH_GRID % column_name

    # Required columns for vector field and fixed point manifests
    required_vector_field_columns = [
        mesh_column_name,
        drift_column_name,
        Column.DATASET,
        Column.SHEAR_STRESS,
    ]
    required_fixed_point_columns = [
        fp_column_name,
        Column.DATASET,
        Column.SHEAR_STRESS,
        Column.FIXED_POINT_STABILITY,
    ]

    # Load drift vector field and fixed points for selected column
    name_suffix = f"_{column_name}_{patch_type}"
    vector_field_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_VECTOR_FIELD}{name_suffix}"
    fixed_points_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix}"
    vector_field_manifest = load_dataframe_manifest(vector_field_manifest_name)
    fixed_points_manifest = load_dataframe_manifest(fixed_points_manifest_name)

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

            vector_field_for_flow_condition = filter_dataframe_by_shear_stress(
                vector_field_dataframe, shear_stress
            )

            drift, centers = get_reshaped_vector_field_and_grid(
                vector_field_for_flow_condition,
                column_names=[column_name],
            )

            fig, ax = plot_drift_1d(
                x_values=centers[-1],
                drift=drift,
                axes_labels=[column_label, f"d{column_label}/dt"],
                figsize=(4, 4),
                drift_line_kwargs={"color": "k", "linewidth": 2},
                zero_line_kwargs={"linestyle": "--", "color": "gray", "linewidth": 1, "alpha": 0.7},
            )
            ax.set_title(fig_title)
            figure_name = f"{dataset_name_flow}{name_suffix}_drift_vector_field"
            save_plot_to_path(fig, output_path, figure_name, file_format=".png")

            if fixed_points_dataframe is not None:
                fixed_points_for_flow_condition = filter_dataframe_by_shear_stress(
                    fixed_points_dataframe, shear_stress
                )
                stable_fixed_points = filter_dataframe_by_stability(
                    fixed_points_for_flow_condition, stability_label=StabilityLabel.STABLE
                )
                ax.plot(
                    stable_fixed_points[fp_column_name],
                    np.zeros_like(stable_fixed_points[fp_column_name]),
                    FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].marker,
                    color=FIXED_POINT_PLOT_STYLE[StabilityLabel.STABLE].color,
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                    markersize=5,
                )
                legend_handles = make_legend_handles_for_fixed_pts([StabilityLabel.STABLE])
                ax.legend(handles=legend_handles, loc="upper right", fontsize="small")
                figure_name = f"{dataset_name_flow}{name_suffix}_stable_fixed_points"
                save_plot_to_path(fig, output_path, figure_name, file_format=".png")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
