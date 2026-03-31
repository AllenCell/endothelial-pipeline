# %%
import logging
from itertools import product
from typing import cast

import numpy as np

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    filter_dataframe_by_annotations,
    get_traj_and_diff,
    split_dataset_by_flow,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    select_bandwidth_cross_validation,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_DYNAMICS,
    BIN_LIMITS_THETA_RESCALED,
    BIN_WIDTHS_DYNAMICS,
    DEFAULT_DATASETS_DYNAMICS_VIS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_NAMES_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
    RESCALE_THETA,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
logger = logging.getLogger(__name__)

# get label for provided feature column
column_names = list(DYNAMICS_COLUMN_NAMES)
crop_pattern = "grid"
variable_labels = [get_label_for_column(col).replace("polar ", "") for col in column_names]
columns_to_compute = [*METADATA_COLUMNS_TO_KEEP[crop_pattern], *column_names]
bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]

# cast global constant dicts to avoid type errors
bin_limits_dict = cast(
    dict[str | ColumnName.DiffAEData, tuple[float, float]], BIN_LIMITS_DYNAMICS.copy()
)

# unpack default bin widths and limits for each column, adjusting limits if
# rescaling theta
if RESCALE_THETA:
    bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE] = BIN_LIMITS_THETA_RESCALED
polar_angle_period = (
    bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][1]
    - bin_limits_dict[ColumnName.DiffAEData.POLAR_ANGLE][0]
)

# get dataframe manifest for crop-based features
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{crop_pattern}"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

# Use provided datasets or default if none provided.
dataset_names = get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)
if DEMO_MODE:
    dataset_names = dataset_names[:2]

# init bandwidth candidates for cross-validation
bandwidth_candidates = np.linspace(0.01, 0.5, 5)
# turn into list of kernel candidates (list of lists, each list represents a
# multivariate product kernel) -- each possible combination of bandwidths across
# the columns will be tested in cross-validation
kernel_candidates = []
for bw_1, bw_2, bw_3 in list(product(bandwidth_candidates, repeat=len(column_names))):
    kernel = []
    for column_name, bw in [
        (column_names[0], bw_1),
        (column_names[1], bw_2),
        (column_names[2], bw_3),
    ]:
        name = KERNEL_NAMES_DYNAMICS[column_name]
        period = polar_angle_period if column_name == ColumnName.DiffAEData.POLAR_ANGLE else None
        kernel.append(KramersMoyalKernel(name=name, bandwidth=bw, period=period))
    kernel_candidates.append(kernel)

# %%
# loop over datasets in collection, compute 1D drift for given variable, and
# plot results, skipping datasets not found in manifest
for dataset_name in dataset_names:
    if dataset_name not in feature_dataframe_manifest.locations:
        logger.warning(
            f"Dataset {dataset_name} not found in manifest {feature_dataframe_manifest_name}. Skipping."
        )
        continue
    dataset_config = load_dataset_config(dataset_name)

    # load dataframe and perform additional filtering (remove
    # non-steady-state timepoints based on annotations), computing
    # only the columns needed for flow field estimation and analysis to save memory.
    df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df_ = df[columns_to_compute].compute()
    df_steady_state = filter_dataframe_by_annotations(
        df_,
        dataset_config,
        timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
    )

    df_by_flow, shear_stress_list = split_dataset_by_flow(df_steady_state, dataset_config)

    # compute on a per-shear stress condition basis
    for df_flow, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
        # get bins for flow field estimation based on the trajectories, to be
        # used for kernel-convolution-based estimation of the Kramers-Moyal
        # coefficients. The bins are determined by the specified bin widths and
        # the range of the data.
        bins, centers = get_bins(
            bin_widths,
            data=df_steady_state[column_names].to_numpy(),
        )
        # get trajectories and differences for the given variable, adjusting
        # polar angle differences for periodicity if needed
        trajectories, displacements = get_traj_and_diff(
            df_, column_names=column_names, polar_angle_period=polar_angle_period
        )

        # select kernel bandwidth by cross-validation
        best_kernel, cv_scores = select_bandwidth_cross_validation(
            trajectories=trajectories,
            displacements=displacements,
            dt=5 / 60,  # time step in hours (1 frame = 5 minutes)
            bins=bins,
            kernel_candidates=kernel_candidates,
            n_jobs=-1,
        )
        print(f"Dataset: {dataset_name}, shear stress: {shear_stress}")
        print(f"   Best kernel: {best_kernel}")
        print(f"   CV scores: {cv_scores}")
# %%
