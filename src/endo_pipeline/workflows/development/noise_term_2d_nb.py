# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
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
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

# %%
logger = logging.getLogger(__name__)

output_path = get_output_path(__file__)

# workflow constants
column_names = [ColumnName.DiffAEData.POLAR_RADIUS, ColumnName.DiffAEData.PC3_FLIPPED]
dataset_names = get_datasets_in_collection("timelapse")
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

        # Assemble the symmetric 2x2 diffusion tensor D at each grid point
        # (component order D_xx, D_yy, D_xy with D_yx = D_xy) and solve
        # D = (1/2) sigma sigma^T for the noise amplitude sigma via the
        # lower-triangular Cholesky factor of 2D.
        diffusion_tensor = np.stack(
            [
                np.stack([diffusion_coeffs[..., 0], diffusion_coeffs[..., 2]], axis=-1),
                np.stack([diffusion_coeffs[..., 2], diffusion_coeffs[..., 1]], axis=-1),
            ],
            axis=-2,
        )
        sigma = np.full_like(diffusion_tensor, np.nan)
        for index in np.ndindex(diffusion_tensor.shape[:-2]):
            tensor = diffusion_tensor[index]
            if not np.all(np.isfinite(tensor)):
                continue
            try:
                sigma[index] = np.linalg.cholesky(2.0 * tensor)
            except np.linalg.LinAlgError:
                # diffusion tensor is not positive semi-definite at this bin
                continue

        # adjust drift term based on the SDE interpretation (e.g., Ito, Stratonovich)
        # spurious drift correction: Delta a_i = sum_{j,k} sigma_kj * d(sigma_ij)/d(x_k)
        # grad_sigma[..., k, i, j] = d(sigma_ij) / d(x_k) using physical bin spacing
        spacings = [center[1] - center[0] for center in centers]
        grad_sigma = np.stack(
            [np.gradient(sigma, spacings[k], axis=k) for k in range(len(centers))],
            axis=-3,
        )
        drift_corrected = drift_coeffs.copy()
        drift_corrected -= SDE_ALPHA * np.einsum("...kj,...kij->...i", sigma, grad_sigma)

        # normalize the drift vector field for visualization purposes
        drift_magnitude = np.linalg.norm(drift_coeffs, axis=-1, keepdims=True)
        drift_coeffs_ = np.divide(
            drift_coeffs,
            drift_magnitude,
            out=np.zeros_like(drift_coeffs),
            where=drift_magnitude != 0,
        )

        drift_corrected_magnitude = np.linalg.norm(drift_corrected, axis=-1, keepdims=True)

        # normalize the corrected drift vector field for visualization purposes
        drift_corrected_ = np.divide(
            drift_corrected,
            drift_corrected_magnitude,
            out=np.zeros_like(drift_corrected),
            where=drift_corrected_magnitude != 0,
        )

        # for each component of the noise amplitude, plot the corresponding drift vector field
        # and the corrected drift vector field for that noise component
        fig, ax = plt.subplots()
        ax.quiver(
            *feature_grid,
            drift_coeffs_[..., 0],
            drift_coeffs_[..., 1],
            color="k",
            scale=10,
            scale_units="xy",
            angles="xy",
        )
        ax.quiver(
            *feature_grid,
            drift_corrected_[..., 0],
            drift_corrected_[..., 1],
            color="red",
            alpha=0.5,  # set transparency for the corrected drift vectors
            scale=10,
            scale_units="xy",
            angles="xy",
        )
        ax.set_xlabel(column_names[0])
        ax.set_ylabel(column_names[1])
        ax.set_title(
            f"Dataset: {dataset_name}, Shear Stress: {shear_stress}, SDE interpretation $\\alpha$: {SDE_ALPHA}"
        )
        plt.show()


# %%
