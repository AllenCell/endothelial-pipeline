from endo_pipeline.cli import Datasets


def main(datasets: Datasets | None = None) -> None:
    """
    Validate fixed points identified in 2D in (r, rho) and 1D in theta against
    the fixed points identified in the full 3D (r, rho, theta) space.

    #validation #fixed-points #dynamics
    """
    import logging
    from typing import cast

    import numpy as np
    import pandas as pd

    from endo_pipeline.io import join_sorted_strings, load_dataframe
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_stability
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.examples import EXAMPLE_DATASET
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        StabilityLabel,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    columns_r_rho = cast(list[str], [Column.DiffAEData.POLAR_RADIUS, Column.DiffAEData.PC3_FLIPPED])
    columns_r_rho_str = join_sorted_strings(columns_r_rho)
    column_theta = Column.DiffAEData.POLAR_ANGLE
    column_names = [column_theta, *columns_r_rho]

    dataset_names = datasets or [
        EXAMPLE_DATASET["FIGURE_2_LOW_FLOW_DATASET"],
        EXAMPLE_DATASET["FIGURE_2_HIGH_FLOW_DATASET"],
    ]

    base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
    fixed_points_r_rho_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{columns_r_rho_str}_{base_name}"
    )
    fixed_points_r_rho_dataframe_manifest = load_dataframe_manifest(
        fixed_points_r_rho_dataframe_manifest_name
    )
    fixed_points_theta_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{column_theta}_{base_name}"
    )
    fixed_points_theta_dataframe_manifest = load_dataframe_manifest(
        fixed_points_theta_dataframe_manifest_name
    )
    bootstrap_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_BOOTSTRAPPING}_{base_name}"
    bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_dataframe_manifest_name)

    absolute_errors: list[float] = []
    for dataset_name in dataset_names:
        if dataset_name not in fixed_points_theta_dataframe_manifest.locations:
            logger.warning(
                "Dataset %s not found in fixed points theta dataframe manifest. Skipping dataset.",
                dataset_name,
            )
            continue
        if dataset_name not in fixed_points_r_rho_dataframe_manifest.locations:
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

        df_fixed_points_theta = load_dataframe(
            fixed_points_theta_dataframe_manifest.locations[dataset_name]
        )
        stable_fixed_points_theta = filter_dataframe_by_stability(
            df_fixed_points_theta, StabilityLabel.STABLE
        )
        df_fixed_points_r_rho = load_dataframe(
            fixed_points_r_rho_dataframe_manifest.locations[dataset_name]
        )
        stable_fixed_points_r_rho = filter_dataframe_by_stability(
            df_fixed_points_r_rho, StabilityLabel.STABLE
        )
        df_2_plus_1d = pd.merge(
            stable_fixed_points_theta,
            stable_fixed_points_r_rho,
            on=[Column.DATASET, Column.SHEAR_STRESS, Column.VectorField.STABILITY],
        )
        df_bootstrap = load_dataframe(bootstrap_dataframe_manifest.locations[dataset_name])
        df_3d_bootstrap = filter_dataframe_by_stability(df_bootstrap, StabilityLabel.STABLE)

        for column_name in column_names:
            coord_2_plus_1d = df_2_plus_1d[column_name].iloc[0]
            coord_3d_bootstrap = df_3d_bootstrap[column_name].iloc[0]
            if column_name == Column.DiffAEData.POLAR_ANGLE:
                # account for periodicity when comparing theta values by unwrapping the angles
                # before computing the error
                unwrapped_2_plus_1d = np.unwrap([coord_2_plus_1d, coord_3d_bootstrap])[0]
                absolute_error = np.abs(unwrapped_2_plus_1d - coord_3d_bootstrap)
            else:
                absolute_error = np.abs(coord_2_plus_1d - coord_3d_bootstrap)

            print(
                f"Dataset: {dataset_name}, Feature: {column_name}, Absolute Error: {absolute_error:.4f}"
            )
            absolute_errors.append(absolute_error)

        print(
            f"Maximum absolute error across all features and datasets: {max(absolute_errors):.4f}"
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
