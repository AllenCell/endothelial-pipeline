# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import (
    filter_dataframe_to_flow_condition_by_timepoint,
    filter_dataframe_to_steady_state,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings.column_names import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    KERNEL_PERIODS_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.flow_field_3d import PAD_BINS_FLOAT
from endo_pipeline.settings.plot_defaults import VECTOR_FIELD_THETA_RANGE
from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

# %%
logger = logging.getLogger(__name__)

output_path = get_output_path(__file__)

# workflow constants
column_names = [ColumnName.DiffAEData.POLAR_ANGLE]
dataset_names = SUMMARY_PLOT_DATASETS["low_high"]
patch_type = "grid_based"
SDE_ALPHA = 0.5

# %%
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

# Initialize kernels and bin widths for each selected column
kernels: list[KramersMoyalKernel] = []
bin_widths: list[float] = []
for column_name in column_names:
    kernels.append(
        KramersMoyalKernel(
            name=KERNEL_NAMES_DYNAMICS[column_name],
            bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[column_name],
            period=KERNEL_PERIODS_DYNAMICS[column_name],
        )
    )
    bin_widths.append(BIN_WIDTHS_DYNAMICS[column_name])

for dataset_name in dataset_names:
    if dataset_name not in feature_dataframe_manifest.locations:
        logger.warning(
            "Dataset '%s' not found in manifest '%s'. Skipping.",
            dataset_name,
            feature_dataframe_manifest_name,
        )
        continue
    dataset_config = load_dataset_config(dataset_name)

    # Load feature dataframe for dataset with only the required columns and
    # filter out non-steady-state timepoints
    df_ = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
    df = df_[columns_to_compute].compute()
    df_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

    for flow_condition in dataset_config.flow_conditions:
        shear_stress = flow_condition.shear_stress
        df_flow = filter_dataframe_to_flow_condition_by_timepoint(
            df_steady_state, dataset_config, flow_condition
        )
        metadata_dict = {
            ColumnName.DATASET: dataset_name,
            ColumnName.SHEAR_STRESS: shear_stress,
        }
        bins, centers = get_bins(
            bin_widths,
            data=df_flow[column_names].to_numpy(),
            pad=PAD_BINS_FLOAT,
        )
        feature_grid = np.meshgrid(*centers, indexing="ij")
        # get list of per-crop trajectories, the corresponding
        # displacement vectors, and time differences
        traj_list, d_traj_list = get_traj_and_diff(df_flow, column_names)

        # get drift and diffusion estimates in units hours^-1 for each bin in 3D
        # space (Kramers-Moyal coefficient estimation)
        drift_coeffs, diffusion_coeffs = get_kramers_moyal_coeffs(
            traj_list, d_traj_list, bins=bins, dt=TIME_STEP_IN_HOURS, kernel=kernels
        )

        # re-wrap theta values to be within the specified x-axis limits for better
        # visualization of the drift as a function of theta
        where_theta_below_xlim = centers[0] < VECTOR_FIELD_THETA_RANGE[0]
        where_theta_above_xlim = centers[0] > VECTOR_FIELD_THETA_RANGE[1]
        theta_values_wrapped = centers[0].copy()
        theta_values_wrapped[where_theta_below_xlim] += np.pi
        theta_values_wrapped[where_theta_above_xlim] -= np.pi
        arg_sorted_theta = np.argsort(theta_values_wrapped)
        theta_values_sorted = theta_values_wrapped[arg_sorted_theta]
        drift_sorted = drift_coeffs[arg_sorted_theta]
        diffusion_sorted = diffusion_coeffs[arg_sorted_theta]

        # compute noise amplitude from diffusion coefficient
        noise_amplitude_sorted = np.sqrt(2 * diffusion_sorted)

        # based on the SDE interpretation (e.g., Ito, Stratonovich), adjust
        # drift term if necessary by subtracting the noise-induced drift term
        drift_corrected = drift_sorted - SDE_ALPHA * noise_amplitude_sorted * np.gradient(
            noise_amplitude_sorted, theta_values_sorted
        )

        fig, ax = plt.subplots()
        ax.plot(theta_values_sorted, drift_sorted, "k-", label="Drift")
        ax.plot(
            theta_values_sorted,
            drift_corrected,
            "r-.",
            label=f"Corrected Drift ($\\alpha = {SDE_ALPHA}$)",
        )
        ax.plot(theta_values_sorted, noise_amplitude_sorted, "b--", label="Noise Amplitude")
        # dashed line at the origin
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7, label="y=0")
        ax.set_xlabel(column_names[0])
        ax.set_xlim(*VECTOR_FIELD_THETA_RANGE)
        ax.set_xticks([0, np.pi / 2], labels=[f"0={Unicode.PI}", f"{Unicode.PI}/2"])
        ax.set_ylabel("Coefficient")
        ax.set_ylim(-1.5, 1.5)
        ax.legend()
        ax.set_title(f"Dataset: {dataset_name}, Shear Stress: {shear_stress}")
        plt.show()


# %%
