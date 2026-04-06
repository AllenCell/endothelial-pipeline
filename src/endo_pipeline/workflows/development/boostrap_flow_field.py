from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    num_bootstrap_iterations: int = 100,
) -> None:
    """Bootstrap fixed point confidence intervals by subsampling data.

    Parameters
    ----------
    crop_pattern
        The crop pattern to use features from.
    datasets
        Optional, specific dataset(s) to run the workflow on.
    num_bootstrap_iterations
        Number of bootstrap iterations to perform for each dataset.

    """
    import logging

    import numpy as np

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe
    from endo_pipeline.library.analyze.bootstrap_fixed_points import (
        run_flow_field_and_fixed_points,
        subsample_trajectories_and_displacements,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
    from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
    from endo_pipeline.manifests import create_dataframe_manifest, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        KERNEL_NAMES_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
        PERIOD_THETA_RESCALED,
        RESCALE_THETA,
    )
    from endo_pipeline.settings.flow_field_3d import (
        DATASET_COLLECTION_FOR_3D_DYNAMICS,
        PAD_BINS_FLOAT,
    )
    from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)
    rng = np.random.default_rng(RANDOM_SEED)

    model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
    run_name = DEFAULT_MODEL_RUN_NAME
    column_names: list[ColumnName.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]

    base_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    dataframe_savedir = get_output_path(__file__, crop_pattern)
    demo_suffix = "_demo" if DEMO_MODE else ""
    baseline_fixed_point_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}{demo_suffix}"
    )
    baseline_fixed_point_manifest = create_dataframe_manifest(
        baseline_fixed_point_manifest_name, workflow_name=__file__
    )
    logger.info("Bootstrap fixed point dataframes will be saved to: [ %s ]", dataframe_savedir)

    dataset_names = datasets or get_datasets_in_collection(DATASET_COLLECTION_FOR_3D_DYNAMICS)
    if DEMO_MODE:
        logger.warning("DEMO MODE: Processing no more than two datasets for quick testing.")
        num_datasets = min(len(dataset_names), 2)
        dataset_names = dataset_names[:num_datasets]

    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    rescaled_theta_period = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    for column_name in column_names:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        bandwidth = KERNEL_BANDWIDTHS_DYNAMICS[column_name]
        period = rescaled_theta_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
        bin_width = BIN_WIDTHS_DYNAMICS[column_name]
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
        bin_widths.append(bin_width)

    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping.",
                feature_dataframe_manifest_name,
                dataset_name,
            )
            continue
        elif dataset_name not in baseline_fixed_point_manifest.locations:
            logger.info(
                "No baseline fixed point dataframe found in manifest [ %s ] for dataset [ %s ]. Skipping.",
                baseline_fixed_point_manifest_name,
                dataset_name,
            )

        dataset_config = load_dataset_config(dataset_name)
        if len(dataset_config.shear_stress_regime) > 1:
            logger.warning(
                "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
                "Skipping for bootstrap 3D flow field analysis.",
                dataset_name,
                dataset_config.shear_stress_regime,
            )
            continue

        # Load the baseline fixed point dataframe for this dataset
        baseline_fp_df = load_dataframe(baseline_fixed_point_manifest.locations[dataset_name])
        logger.debug(
            "Number of baseline fixed points for dataset [ %s ]: [ %d ]",
            dataset_name,
            len(baseline_fp_df),
        )

        # Load and filter the feature dataframe to steady-state timepoints
        # (will use for bootstrap iterations)
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # Determine bins from the full steady-state data (shared across all
        # bootstrap iterations so the fixed-point search uses a consistent grid)
        bins, centers = get_bins(
            bin_widths,
            data=df_steady_state[column_names].to_numpy(),
            pad=PAD_BINS_FLOAT,
        )

        # Compute trajectories and displacements once from the full steady-state
        # data; the same lists are reused for the baseline and subsampled for
        # each bootstrap iteration.
        full_trajectories, full_displacements = get_traj_and_diff(df_steady_state, column_names)

        # ---- Begin bootstrap loop here ----
        for _ in range(num_bootstrap_iterations):
            # Subsample trajectories and displacements for this iteration
            subsampled_trajectories, subsampled_displacements = (
                subsample_trajectories_and_displacements(
                    full_trajectories, full_displacements, subsample_fraction=0.5, rng=rng
                )
            )

            # Run the flow field and fixed point pipeline on the subsampled data
            # (using the full steady-state dataframe for bounds computation in
            # fixed point finding, so that bounds reflect the true data
            # distribution even in bootstrap iterations)
            fixed_point_df_from_subsample = run_flow_field_and_fixed_points(
                subsampled_trajectories,
                subsampled_displacements,
                df_for_bounds=df_steady_state,
                bins=bins,
                centers=centers,
                column_names=column_names,
                kernels=kernels,
            )
            print(fixed_point_df_from_subsample.head())


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
