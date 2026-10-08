import logging
from itertools import product

import pandas as pd

from endo_pipeline.cli import DEMO_MODE, UPLOAD_TO_FMS, Datasets, PatchType, StrList
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    join_sorted_strings,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.vector_field_estimation import (
    get_drift_estimates_and_fixed_points,
    get_valid_flow_field_column_names,
)
from endo_pipeline.manifests import (
    DataframeLocation,
    create_dataframe_manifest,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.bandwidth_binwidth_sweep import (
    BINWIDTH_EXPONENT_LIMITS,
    KERNEL_BANDWIDTH_EXPONENT_LIMITS,
    PARAMETER_SCALE_BASE,
)
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    KERNEL_PERIODS_DYNAMICS,
    LOWER_PERCENTILE_FOR_FILTERING_FPTS,
    METADATA_COLUMNS_TO_KEEP,
    NUM_INIT_SAMPLES,
    TIME_STEP_IN_HOURS,
    UPPER_PERCENTILE_FOR_FILTERING_FPTS,
)
from endo_pipeline.settings.flow_field_dataframes import (
    FMS_ANNOTATION_NOTES_FIXED_POINTS,
    FMS_ANNOTATION_NOTES_VECTOR_FIELD,
)
from endo_pipeline.settings.manifest_names import (
    FIXED_POINT_MANIFEST_NAMES,
    VECTOR_FIELD_MANIFEST_NAMES,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
    RANDOM_SEED,
)


def get_param_sweep_run_name(bin_scale: int, kernel_scale: int) -> str:
    """
    Generate a name for the parameter sweep based on the bin and kernel scales.

    Parameters
    ----------
    bin_scale
        Exponent used to scale the default bin widths.
    kernel_scale
        Exponent used to scale the default kernel bandwidths.

    Returns
    -------
    :
        Name representing the parameter sweep.
    """
    tag = f"bwScale{bin_scale}_kbScale{kernel_scale}"
    run_name_for_sweep_condition = f"{DEFAULT_MODEL_RUN_NAME}_{tag}"

    return run_name_for_sweep_condition


def apply_parameter_scaling(
    bin_scale: int, kernel_scale: int, param_scale_base: int = PARAMETER_SCALE_BASE
) -> tuple[dict[Column.DiffAEData, float], dict[Column.DiffAEData, float]]:
    """
    Apply scaling exponents to the default bin widths and kernel bandwidths.

    Parameters
    ----------
    bin_scale
        Exponent to scale the default bin widths.
    kernel_scale
        Exponent to scale the default kernel bandwidths.
    param_scale_base
        Base of the exponent for scaling, by default `PARAMETER_SCALE_BASE`

    Returns
    -------
    :
        Scaled bin widths and kernel bandwidths.
    """

    bin_widths = {
        key: val * param_scale_base**bin_scale for key, val in BIN_WIDTHS_DYNAMICS.items()
    }
    kernel_bandwidths = {
        key: val * param_scale_base**kernel_scale for key, val in KERNEL_BANDWIDTHS_DYNAMICS.items()
    }
    return bin_widths, kernel_bandwidths


def get_parameter_space_scalings() -> list[tuple[int, int]]:
    """
    Get the parameter space of bin width and kernel bandwidth scaling exponents.

    Returns
    -------
    :
        (bin width, kernel bandwidth) combinations of the exponents used for scaling default values.
    """

    # we will scale the default parameters by exponents of base 2
    # (i.e. parameters will be quarter, half, original size, double, quadruple, etc.)
    binwidth_exponent_range: list[int] = list(
        range(BINWIDTH_EXPONENT_LIMITS[0], BINWIDTH_EXPONENT_LIMITS[1] + 1)
    )
    kernel_bandwidth_exponent_range: list[int] = list(
        range(KERNEL_BANDWIDTH_EXPONENT_LIMITS[0], KERNEL_BANDWIDTH_EXPONENT_LIMITS[1] + 1)
    )

    # create the full parameter space of these scaling exponents, which will then be applied to the
    # default bin widths and kernel bandwidths used in the paper
    parameter_space_scaling = list(
        product(binwidth_exponent_range, kernel_bandwidth_exponent_range)
    )

    return parameter_space_scaling


def make_combined_filtered_feature_dataframe_for_dynamics_workflows(
    dataset_name_1: str, dataset_name_2: str, patch_type: PatchType = "grid_based"
) -> pd.DataFrame:
    # Workflow only supports generating flow fields from combinations of
    # three specific column names (as defined in DYNAMICS_COLUMN_NAMES). If
    # column names are provided, ensure they are a subset of these columns and
    # skip any that are not. If no column names are provided, just use these
    # three columns.
    columns = None
    column_names = get_valid_flow_field_column_names(columns)

    # Columns to keep when loading feature dataframe
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    control_df = pd.DataFrame()
    for dataset_name in (dataset_name_1, dataset_name_2):
        dataset_config = load_dataset_config(dataset_name)
        # Load feature dataframe for dataset with only the required columns and
        # filter out non-steady-state timepoints
        df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
        df = df_[columns_to_compute].compute()
        df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

        # # Generate vector field and calculate fixed points per flow condition
        # vector_field_dataframe_list = []
        # fixed_points_dataframe_list = []

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                df_steady_state, dataset_config, flow_condition
            )

            df_flow_sample_size_for_half = len(df_flow) // 2
            df_flow_subsample = df_flow.sample(
                n=df_flow_sample_size_for_half, random_state=RANDOM_SEED
            )
            control_df = pd.concat([control_df, df_flow_subsample], ignore_index=True)

    return control_df


def generate_flow_field_for_low_high_control(
    patch_type: PatchType = "grid_based",
    columns: StrList | None = None,
    datasets: Datasets | None = None,
    sweep_name: str | None = None,
    kernel_bandwidths_dynamics: dict[Column.DiffAEData, float] | None = None,
    bin_widths_dynamics: dict[Column.DiffAEData, float] | None = None,
) -> None:
    """
    Generate drift vector field and estimate fixed points.

    #dynamical-systems #fixed-points #grid-based #cell-centered #test-ready

    This workflow generates the flow field based on the following features
    derived from evaluating the DiffAE model on specific patches of the data:

    - `polar_theta` = polar angle coordinate computed from PC1 and PC2
    - `polar_r` = polar radius coordinate computed from PC1 and PC2
    - `rho` = PC3 value with sign flipped

    Any combination of these features can be used to generate the flow field
    with the corresponding dimensionality. By default, the workflow will
    generate the a 3D flow field using all three features.

    Using the feature space defined by the specified input features, this
    workflow does the following for each specified dataset:

    1. Estimate drift flow fields using a kernel-based method for estimating
       Kramers-Moyal coefficients from time series data
    2. Use interpolation to get a callable flow field function
    3. Identify stable fixed points of the flow field using a root-finding
       method applied to the flow field function
    4. Save the following outputs for each dataset as parquet files:
        - Dataframe with the drift vector field
        - Dataframe with the drift fixed points

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe generate-flow-field -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe generate-flow-field --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will generate the
    flow field for the first dataset.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    columns
        Specific columns to use to generate flow field.
    datasets
        List of datasets or dataset collections to generate flow fields for.
    run_name
        Optional name for the current run, used for organizing output files.
    """

    import pandas as pd

    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.library.analyze.vector_field_estimation import (
        get_valid_flow_field_column_names,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        create_dataframe_manifest,
        load_dataframe_manifest,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        KERNEL_BANDWIDTHS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.manifest_names import FIXED_POINT_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    if sweep_name is None:
        sweep_name = ""

    if kernel_bandwidths_dynamics is None:
        kernel_bandwidths_dynamics = KERNEL_BANDWIDTHS_DYNAMICS

    if bin_widths_dynamics is None:
        bin_widths_dynamics = BIN_WIDTHS_DYNAMICS

    dataset_names = datasets or ["20250618_20X", "20250611_20X"]

    # Workflow only supports generating flow fields from combinations of
    # three specific column names (as defined in DYNAMICS_COLUMN_NAMES). If
    # column names are provided, ensure they are a subset of these columns and
    # skip any that are not. If no column names are provided, just use these
    # three columns.
    column_names = get_valid_flow_field_column_names(columns)

    if not column_names:
        logger.error("No valid columns for generating flow field.")
        return

    logger.info("Generating flow field for columns: %s", column_names)

    # Columns to keep when loading feature dataframe
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Load default model manifest and corresponding feature dataframe for
    # specified patch type
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # Build dataframe manifest names that include sorted list of selected
    # columns used to generate the flow field.
    name_suffix = join_sorted_strings(column_names)
    vector_field_dataframe_manifest_name = "_".join(
        filter(None, [VECTOR_FIELD_MANIFEST_NAMES[patch_type], name_suffix, sweep_name])
    )
    fixed_points_dataframe_manifest_name = "_".join(
        filter(None, [FIXED_POINT_MANIFEST_NAMES[patch_type], name_suffix, sweep_name])
    )
    vector_field_dataframe_manifest = create_dataframe_manifest(
        vector_field_dataframe_manifest_name, workflow_name=__file__
    )
    fixed_points_dataframe_manifest = create_dataframe_manifest(
        fixed_points_dataframe_manifest_name, workflow_name=__file__
    )

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    for column_name in column_names:
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=kernel_bandwidths_dynamics[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(bin_widths_dynamics[column_name])

    # Add parameters to dataframe manifests for traceability
    for output_dataframe_manifest in [
        vector_field_dataframe_manifest,
        fixed_points_dataframe_manifest,
    ]:
        output_dataframe_manifest.parameters = {
            "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
            "run_name": DEFAULT_MODEL_RUN_NAME,
            "patch_type": patch_type,
            "kernels": [
                {
                    "column": str(column),
                    "name": kernel.name,
                    "bandwidth": kernel.bandwidth,
                    "period": kernel.period,
                }
                for column, kernel in zip(column_names, kernels, strict=False)
            ],
            "bin_widths": bin_widths,
            "num_init_samples_for_root_solver": NUM_INIT_SAMPLES,
            "lower_percentile_for_filtering_fpts": LOWER_PERCENTILE_FOR_FILTERING_FPTS,
            "upper_percentile_for_filtering_fpts": UPPER_PERCENTILE_FOR_FILTERING_FPTS,
        }
        save_dataframe_manifest(output_dataframe_manifest)

    df_flow = make_combined_filtered_feature_dataframe_for_dynamics_workflows(
        dataset_name_1=dataset_names[0], dataset_name_2=dataset_names[1], patch_type=patch_type
    )

    # for dataset_name in dataset_names:
    #     if dataset_name not in feature_dataframe_manifest.locations:
    #         logger.warning(
    #             "Dataset '%s' not found in manifest '%s'. Skipping.",
    #             dataset_name,
    #             feature_dataframe_manifest_name,
    #         )
    #         continue

    #     dataset_config = load_dataset_config(dataset_name)

    #     # Load feature dataframe for dataset with only the required columns and
    #     # filter out non-steady-state timepoints
    #     df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    #     df = df_[columns_to_compute].compute()
    #     df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    # Generate vector field and calculate fixed points per flow condition
    vector_field_dataframe_list = []
    fixed_points_dataframe_list = []
    shear_stress_list = []
    dataset_config_list = []

    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)
        dataset_config_list.append(dataset_config)

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            shear_stress_list.append(shear_stress)

    dataset_name = "-".join(dataset_names)
    shear_stress = "-".join(map(str, shear_stress_list))
    metadata_dict: dict[str, str | float] = {
        ColumnName.DATASET: dataset_name,
        ColumnName.SHEAR_STRESS: shear_stress,
    }
    vector_field_dataframe, fixed_points_dataframe = get_drift_estimates_and_fixed_points(
        dataframe=df_flow,
        column_names=column_names,
        bin_widths=bin_widths,
        kernel=kernels,
        time_step=TIME_STEP_IN_HOURS,
        metadata_dict=metadata_dict,
    )

    # Append vector field dataframe to list to be concatenated outside
    # of flow condition loop
    vector_field_dataframe_list.append(vector_field_dataframe)

    # Append fixed points dataframe to list to be concatenated outside
    # of flow condition loop, if the dataframe is not empty
    if not fixed_points_dataframe.empty:
        fixed_points_dataframe_list.append(fixed_points_dataframe)

    # Concatenate vector fields for flow conditions into single dataframe.
    vector_field_for_dataset = pd.concat(vector_field_dataframe_list, ignore_index=True)

    # If there are any fixed points for the dataset, concatenate them into
    # a single dataframe.
    if fixed_points_dataframe_list:
        fixed_points_for_dataset = pd.concat(fixed_points_dataframe_list, ignore_index=True)
    else:
        fixed_points_for_dataset = None

    for manifest, dataframe, name_prefix, additional_notes in [
        (
            vector_field_dataframe_manifest,
            vector_field_for_dataset,
            VECTOR_FIELD_MANIFEST_NAMES[patch_type],
            FMS_ANNOTATION_NOTES_VECTOR_FIELD % len(column_names),
        ),
        (
            fixed_points_dataframe_manifest,
            fixed_points_for_dataset,
            FIXED_POINT_MANIFEST_NAMES[patch_type],
            FMS_ANNOTATION_NOTES_FIXED_POINTS % len(column_names),
        ),
    ]:
        # Skip if dataframe is None.
        if dataframe is None:
            continue

        # Save dataframe to file
        output_filename = "_".join(
            filter(None, [name_prefix, dataset_name, name_suffix, sweep_name])
        )
        save_path = output_path / f"{output_filename}.parquet"
        dataframe.to_parquet(save_path, index=False)

        # Create location object with output path
        location = manifest.locations.get(dataset_name, DataframeLocation())
        location.path = save_path

        # Upload to FMS (internal only) and replace local path with file id
        if UPLOAD_TO_FMS:
            annotations = build_fms_annotations(
                dataset_config_list,
                model_manifest=model_manifest,
                run_name=DEFAULT_MODEL_RUN_NAME,
                additional_notes=additional_notes,
            )
            fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
            location.fmsid = fmsid
            location.path = None

        # Add dataframe location to dataframe manifest and save
        manifest.locations[dataset_name] = location
        save_dataframe_manifest(manifest)


def bootstrap_fixed_points_for_low_high_control(
    patch_type: PatchType = "grid_based",
    datasets: Datasets | None = None,
    num_bootstrap_iterations: Annotated[
        int, Parameter(name="--num-iterations")
    ] = NUM_BOOTSTRAP_ITERATIONS,
    bootstrap_match_radius: Annotated[
        float, Parameter(name="--match-dist")
    ] = BOOTSTRAP_MATCH_RADIUS,
    bootstrap_ci_lower_percentile: Annotated[
        float, Parameter(name="--ci-lower")
    ] = FP_CI_LOWER_PERCENTILE,
    bootstrap_ci_upper_percentile: Annotated[
        float, Parameter(name="--ci-upper")
    ] = FP_CI_UPPER_PERCENTILE,
    batch_size_factor: float = BATCH_SIZE_SCALING_FACTOR,
    sweep_name: str | None = None,
    kernel_bandwidths_dynamics: dict[Column.DiffAEData, float] | None = None,
    bin_widths_dynamics: dict[Column.DiffAEData, float] | None = None,
) -> None:
    """
    Bootstrap fixed point confidence intervals by subsampling data.

    #dynamical-systems #fixed-points #grid-based #cell-centered #test-ready #workers

    For each bootstrap iteration, baseline fixed points are processed in row
    order and each is offered the closest unassigned bootstrap fixed point that
    lies within `BOOTSTRAP_MATCH_RADIUS`.  Each bootstrap fixed point can be
    matched to at most one baseline fixed point per iteration. Iterations that
    yield no fixed points, or no fixed points within radius of a given baseline
    fixed point, are counted as misses for that baseline fixed point.

    Each dataframe contains one row per baseline fixed point, with columns for:

    - `dataset` = dataset identifier
    - `stability` = stability classification of the baseline fixed point
    - `{col}` = baseline coordinate for each feature column
    - `{col}_ci_lower` = lower bootstrap CI bounds for each coordinate at
      percentile `bootstrap_ci_lower_percentile`
    - `{col}_ci_upper` = upper bootstrap CI bounds for each coordinate at
      percentiles `bootstrap_ci_upper_percentile`
    - `bootstrap_detection_rate` =  fraction of bootstrap samples in which a
      matched fixed point was found within `bootstrap_match_radius`
    - `n_bootstrap_samples` = number of bootstrap iterations performed

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe bootstrap-fixed-points -d
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe bootstrap-fixed-points --datasets DATASET_NAME
    ```

    ## Parallel processing

    The bootstrap iterations are parallelized across CPU cores using based on
    requested number of worker processes. The number of bootstrap iterations
    assigned to each worker at a time is determined by the `batch_size_factor`
    parameter, which scales the number of batches relative to the number of
    workers. A smaller `batch_size_factor` (i.e., larger batch size) reduces
    overhead from queuing tasks but may lead to less even load balancing if the
    time per iteration is variable.

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will perform
    bootstrapping on the first dataset with at most 10 bootstrap iterations.

    Parameters
    ----------
    patch_type
        Patch type used to calculate the features.
    datasets
        List of datasets or dataset collections to bootstrap fixed points for.
    num_bootstrap_iterations
        Number of bootstrap iterations to perform for each dataset.
    bootstrap_match_radius
        Maximum distance in feature space for a bootstrap fixed point to be
        considered a match to a given baseline fixed point in each iteration.
    bootstrap_ci_lower_percentile
        Percentile defining lower bound of the bootstrap confidence intervals.
    bootstrap_ci_upper_percentile
        Percentile defining upper bound of the bootstrap confidence intervals.
    batch_size_factor
        Factor used to determine size of batch for parallel processing.
    run_name
        Optional name for the current run, used for organizing output files.
    """

    import os
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from endo_pipeline.cli import NUM_WORKERS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import load_dataframe, make_name_unique
    from endo_pipeline.library.analyze.bootstrap_fixed_points import (
        aggregate_bootstrapping_results,
        init_bootstrap_worker,
        match_bootstrap_fixed_points_to_baseline,
        run_one_bootstrap_iteration,
        sample_trajectories_and_displacements_for_bootstrapping,
    )
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT
    from endo_pipeline.settings.flow_field_dataframes import FMS_ANNOTATION_NOTES_BOOTSTRAPPING
    from endo_pipeline.settings.manifest_names import BOOTSTRAPPING_MANIFEST_NAMES
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_RUN_NAME,
        FEATURES_FILTERED_MANIFEST_NAMES,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)

    output_path = get_output_path(__file__)

    if sweep_name is None:
        sweep_name = ""

    if kernel_bandwidths_dynamics is None:
        kernel_bandwidths_dynamics = KERNEL_BANDWIDTHS_DYNAMICS

    if bin_widths_dynamics is None:
        bin_widths_dynamics = BIN_WIDTHS_DYNAMICS

    rng = np.random.default_rng(RANDOM_SEED)

    column_names = list(DYNAMICS_COLUMN_NAMES)
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[patch_type], *column_names]

    # Get feature dataframe manifest for select grid pattern
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[patch_type]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # get dataframe manifest for baseline results to match against in bootstrapping
    name_suffix = join_sorted_strings(column_names)
    baseline_fixed_point_manifest_name = "_".join(
        filter(None, [FIXED_POINT_MANIFEST_NAMES[patch_type], name_suffix, sweep_name])
    )
    baseline_fixed_point_manifest = load_dataframe_manifest(baseline_fixed_point_manifest_name)

    # load or initialize dataframe manifest for bootstrap results
    name_prefix = BOOTSTRAPPING_MANIFEST_NAMES[patch_type]
    name_suffix = "demo" if DEMO_MODE else ""
    bootstrap_results_manifest_name = "_".join(filter(None, [name_prefix, name_suffix, sweep_name]))
    bootstrap_results_manifest = create_dataframe_manifest(
        bootstrap_results_manifest_name, workflow_name=__file__
    )

    dataset_names = datasets or ["20250618_20X", "20250611_20X"]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting bootstrap iterations to <= 10")
        num_bootstrap_iterations = min(num_bootstrap_iterations, 10)

    # Initialize kernels and bin widths for each selected column
    kernels: list[KramersMoyalKernel] = []
    bin_widths: list[float] = []
    for column_name in column_names:
        kernels.append(
            KramersMoyalKernel(
                name=KERNEL_NAMES_DYNAMICS[column_name],
                bandwidth=kernel_bandwidths_dynamics[column_name],
                period=KERNEL_PERIODS_DYNAMICS[column_name],
            )
        )
        bin_widths.append(bin_widths_dynamics[column_name])

    # Add workflow parameters to the output manifest for traceability
    bootstrap_results_manifest.parameters = {
        "model_manifest_name": DEFAULT_MODEL_MANIFEST_NAME,
        "run_name": DEFAULT_MODEL_RUN_NAME,
        "patch_type": patch_type,
        "kernels": [
            {
                "column": str(column),
                "name": kernel.name,
                "bandwidth": kernel.bandwidth,
                "period": kernel.period,
            }
            for column, kernel in zip(column_names, kernels, strict=False)
        ],
        "bin_widths": bin_widths,
        "num_bootstrap_iterations": num_bootstrap_iterations,
    }
    save_dataframe_manifest(bootstrap_results_manifest)

    df_flow = make_combined_filtered_feature_dataframe_for_dynamics_workflows(
        dataset_name_1=dataset_names[0], dataset_name_2=dataset_names[1], patch_type=patch_type
    )

    shear_stress_list = []
    dataset_config_list = []

    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)
        dataset_config_list.append(dataset_config)

        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress
            shear_stress_list.append(shear_stress)

    dataset_name = "-".join(dataset_names)
    shear_stress = "-".join(map(str, shear_stress_list))
    metadata_dict: dict[str, str | float] = {
        ColumnName.DATASET: dataset_name,
        ColumnName.SHEAR_STRESS: shear_stress,
    }

    # for dataset_name in dataset_names:
    #     if dataset_name not in feature_dataframe_manifest.locations:
    #         logger.warning(
    #             "Dataset '%s' not found in manifest '%s'. Skipping.",
    #             dataset_name,
    #             feature_dataframe_manifest_name,
    #         )
    #         continue

    #     if dataset_name not in baseline_fixed_point_manifest.locations:
    #         logger.warning(
    #             "Dataset '%s' not found in manifest '%s'. Skipping.",
    #             dataset_name,
    #             feature_dataframe_manifest_name,
    #         )
    #         continue

    #     dataset_config = load_dataset_config(dataset_name)

    # Load the baseline fixed point dataframe for this dataset
    baseline_fp_df = load_dataframe(baseline_fixed_point_manifest.locations[dataset_name])
    logger.debug(
        "Number of baseline fixed points for dataset [ %s ]: [ %d ]",
        dataset_name,
        len(baseline_fp_df),
    )
    bootstrap_dataframe_list = []
    # # Load and filter the feature dataframe to steady-state timepoints
    # # (will use for bootstrap iterations)
    # df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    # df = df_[columns_to_compute].compute()
    # df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    # bootstrap_dataframe_list = []
    # for flow_condition in dataset_config.flow_conditions:
    #     shear_stress = flow_condition.shear_stress
    #     df_flow = filter_dataframe_to_flow_condition_by_timepoint(
    #         df_steady_state, dataset_config, flow_condition
    #     )
    #     metadata_dict = {
    #         Column.DATASET: dataset_name,
    #         Column.SHEAR_STRESS: shear_stress,
    #     }
    # fixed_points_for_flow_condition = filter_dataframe_by_shear_stress(
    #     baseline_fp_df, shear_stress
    # )

    # Determine bins from the full steady-state data (shared across all
    # bootstrap iterations so the fixed-point search uses a consistent grid)
    bins, centers = get_bins(
        bin_widths,
        data=df_flow[column_names].to_numpy(),
        pad=PAD_BINS_FLOAT,
    )

    # Compute trajectories and displacements once from the full steady-state
    # data; the same lists are reused for the baseline and subsampled for
    # each bootstrap iteration.
    trajectories, displacements = get_traj_and_diff(df_flow, column_names)

    # Generate all resampled lists of trajectories and displacements for the
    # bootstrap iterations up front, to avoid overhead from repeatedly
    # resampling in each worker process. Each element of `all_sampled_pairs`
    # is a tuple of (resampled_trajectories, resampled_displacements) for
    # one bootstrap iteration.
    all_sampled_pairs: list[tuple[list[np.ndarray], list[np.ndarray]]] = [
        sample_trajectories_and_displacements_for_bootstrapping(
            trajectories, displacements, rng=rng
        )
        for _ in range(num_bootstrap_iterations)
    ]

    # Determine worker and per-worker BLAS thread counts that together
    # stay within the 50% of the CPUs allocated by SLURM (or the OS).
    try:
        n_available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # Windows
        n_available_cpus = os.cpu_count() or 1

    n_available_cpus = max(1, n_available_cpus // 2)
    n_workers = NUM_WORKERS or n_available_cpus
    blas_threads_per_worker = max(1, n_available_cpus // n_workers)

    # Choose a chunksize that avoids both excessive queue overhead (too
    # small) and uneven load balancing (too large).
    batch_size = max(1, num_bootstrap_iterations // (n_workers * batch_size_factor))
    logger.info(
        "Running %d bootstrap iterations for dataset [ %s ] "
        "with %d worker process(es), %d BLAS thread(s) per worker, batch size %d.",
        num_bootstrap_iterations,
        dataset_name,
        n_workers,
        blas_threads_per_worker,
        batch_size,
    )

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_bootstrap_worker,
        initargs=(
            df_flow,
            bins,
            centers,
            column_names,
            kernels,
            blas_threads_per_worker,
        ),
    ) as executor:
        bootstrap_fixed_points: list[pd.DataFrame] = list(
            tqdm(
                executor.map(run_one_bootstrap_iteration, all_sampled_pairs, chunksize=batch_size),
                total=num_bootstrap_iterations,
                desc=f"Bootstrap iterations for dataset: {dataset_name}",
            )
        )

    # num iterations with fixed points = number of dataframes in
    # `bootstrap_fixed_points` with at least one row
    n_iterations_with_fpts = sum(len(result_df) > 0 for result_df in bootstrap_fixed_points)
    logger.info(
        "Bootstrap complete for dataset [ %s ]: %d / %d iterations yielded fixed points.",
        dataset_name,
        n_iterations_with_fpts,
        num_bootstrap_iterations,
    )

    # Aggregate bootstrap results by matching fixed points across iterations
    # to the baseline fixed points and computing confidence intervals and
    # detection rates for each baseline fixed point
    matched_coords_flow = match_bootstrap_fixed_points_to_baseline(
        baseline_fixed_points=baseline_fp_df,
        bootstrap_fixed_points=bootstrap_fixed_points,
        column_names=column_names,
        bootstrap_match_radius=bootstrap_match_radius,
    )
    bootstrap_results_df_flow = aggregate_bootstrapping_results(
        baseline_fixed_points=baseline_fp_df,
        matched_coords=matched_coords_flow,
        column_names=column_names,
        n_bootstrap=num_bootstrap_iterations,
        bootstrap_ci_lower_percentile=bootstrap_ci_lower_percentile,
        bootstrap_ci_upper_percentile=bootstrap_ci_upper_percentile,
        metadata_dict=metadata_dict,
    )
    bootstrap_dataframe_list.append(bootstrap_results_df_flow)

    # Concatenate results across flow conditions for this dataset
    bootstrap_results_df = pd.concat(bootstrap_dataframe_list, ignore_index=True)
    # Save results, upload to FMS (if specified), and update manifest
    output_file_name = "_".join(filter(None, [name_prefix, dataset_name, name_suffix]))
    output_file_name_unique = make_name_unique(f"{output_file_name}.parquet")
    output_save_path = output_path / Path(
        "_".join(filter(None, [output_file_name_unique.stem, sweep_name]))
        + output_file_name_unique.suffix
    )
    bootstrap_results_df.to_parquet(output_save_path)
    logger.info("Saved bootstrap fixed point CI dataframe locally to [ %s ].", output_save_path)

    # Create location object with output path
    location = bootstrap_results_manifest.locations.get(dataset_name, DataframeLocation())
    location.path = output_save_path

    # Upload to FMS (internal only) and replace local path with file id
    if UPLOAD_TO_FMS:
        annotations = build_fms_annotations(
            dataset_config,
            model_manifest=load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
            run_name=DEFAULT_MODEL_MANIFEST_NAME,
            additional_notes=FMS_ANNOTATION_NOTES_BOOTSTRAPPING,
        )
        fmsid = upload_file_to_fms(output_save_path, annotations=annotations, file_type="parquet")
        location.fmsid = fmsid
        location.path = None

    # Add dataframe location to dataframe manifest and save
    bootstrap_results_manifest.locations[dataset_name] = location
    save_dataframe_manifest(bootstrap_results_manifest)
