from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Validate fixed points identified in 2D + 1D versus 3D.

    #validation #fixed-points #dynamics

    Validate fixed points as identified in 2D in (r, rho) and 1D in theta
    against the fixed points identified in the full 3D (r, rho, theta) space.
    This workflow computes the absolute error between the coordinates of the
    fixed points identified in 2D + 1D and the coordinates of the fixed points
    identified in 3D, and prints the results.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-fixed-points -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compare-fixed-points --datasets DATASET_NAME
    ```

    ## Default datasets

    If datasets are not provided, the workflow will run on the datasets
    specified in the EXAMPLE_DATASET dictionary in
    `endo_pipeline.settings.examples` under the keys "FIGURE_2_LOW_FLOW_DATASET"
    and "FIGURE_2_HIGH_FLOW_DATASET".

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will only compare
    fixed points for a single dataset.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to validate.
    """

    import logging

    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.column_names import ColumnNameSuffix
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.flow_field_dataframes import StabilityLabel
    from endo_pipeline.settings.manifest_names import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME,
    )

    logger = logging.getLogger(__name__)

    column_names_2d = [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED]
    column_name_1d = Column.DiffAEData.POLAR_ANGLE
    column_names = [column_name_1d, *column_names_2d]

    dataset_names = datasets or [
        EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"],
        EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"],
    ]

    if DEMO_MODE:
        logger.warning("DEMO_MODE - Limiting to one dataset")
        dataset_names = dataset_names[:1]

    name_suffix_2d = f"_{join_sorted_strings(column_names_2d)}_grid"
    fixed_points_2d_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix_2d}"
    fixed_points_2d_manifest = load_dataframe_manifest(fixed_points_2d_manifest_name)

    name_suffix_1d = f"_{column_name_1d}_grid"
    fixed_points_1d_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix_1d}"
    fixed_points_1d_manifest = load_dataframe_manifest(fixed_points_1d_manifest_name)

    bootstrap_dataframe_manifest = load_dataframe_manifest(GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME)

    absolute_errors: list[float] = []
    for dataset_name in dataset_names:
        if dataset_name not in fixed_points_1d_manifest.locations:
            logger.warning(
                "Dataset %s not found in fixed points theta dataframe manifest. Skipping dataset.",
                dataset_name,
            )
            continue
        if dataset_name not in fixed_points_2d_manifest.locations:
            logger.warning(
                "Dataset %s not found in fixed points r-rho dataframe manifest. Skipping dataset.",
                dataset_name,
            )
            continue
        if dataset_name not in bootstrap_dataframe_manifest.locations:
            logger.warning(
                "Dataset %s not found in bootstrap dataframe manifest. Skipping dataset.",
                dataset_name,
            )
            continue

        df_fixed_points_1d = load_dataframe(fixed_points_1d_manifest.locations[dataset_name])
        stable_fixed_points_1d = filter_dataframe_by_stability(
            df_fixed_points_1d, StabilityLabel.STABLE
        )
        df_fixed_points_2d = load_dataframe(fixed_points_2d_manifest.locations[dataset_name])
        stable_fixed_points_2d = filter_dataframe_by_stability(
            df_fixed_points_2d, StabilityLabel.STABLE
        )
        df_2d_plus_1d = pd.merge(
            stable_fixed_points_1d,
            stable_fixed_points_2d,
            on=[Column.DATASET, Column.SHEAR_STRESS, Column.FIXED_POINT_STABILITY],
        )
        df_bootstrap = load_dataframe(bootstrap_dataframe_manifest.locations[dataset_name])
        df_3d_bootstrap = filter_dataframe_by_stability(df_bootstrap, StabilityLabel.STABLE)

        for column_name in column_names:
            column_name_fixed_point = f"{column_name}{ColumnNameSuffix.FIXED_POINTS}"
            column_name_baseline = f"{column_name}{ColumnNameSuffix.BASELINE_FIXED_POINTS}"
            coord_2d_plus_1d = df_2d_plus_1d[column_name_fixed_point].iloc[0]
            coord_3d_bootstrap = df_3d_bootstrap[column_name_baseline].iloc[0]
            if column_name == Column.DiffAEData.POLAR_ANGLE:
                # account for periodicity when comparing theta values by unwrapping the angles
                # before computing the error
                unwrapped_2d_plus_1d = np.unwrap([coord_2d_plus_1d, coord_3d_bootstrap])[0]
                absolute_error = np.abs(unwrapped_2d_plus_1d - coord_3d_bootstrap)
            else:
                absolute_error = np.abs(coord_2d_plus_1d - coord_3d_bootstrap)

            print(
                f"Dataset: {dataset_name}, "
                f"Feature: {column_name}, "
                f"Absolute Error: {absolute_error:.4f}"
            )
            absolute_errors.append(absolute_error)

    print(f"Maximum absolute error across all features and datasets: {max(absolute_errors):.4f}")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
