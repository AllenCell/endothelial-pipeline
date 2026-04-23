# %%
"""Supplementary figure detailing computation of the drift vector fields from grid-based crop trajectories."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_steady_state
from endo_pipeline.library.analyze.kramers_moyal.km_computation import (
    _check_and_adjust_km_inputs,
    _evaluate_multivariate_product_kernel,
    _get_km_powers,
    _get_weighted_histogram_for_convolution,
    get_cartesian_product,
    get_kramers_moyal_coeffs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.visualize.flow_field_schematic import make_real_image_panel
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    METADATA_COLUMNS_TO_KEEP,
    TIME_STEP_IN_HOURS,
)
from endo_pipeline.settings.flow_field_2d import (
    DRIFT_CONTOUR_COLORMAP,
    DRIFT_CONTOUR_VMAX,
    DRIFT_CONTOUR_VMIN,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
)

# %%
plt.style.use("endo_pipeline.figure")


output_path = get_output_path("supp_fig_flow_field")
# %%
image_panel_path = make_real_image_panel(output_path)

# %%
# Use the dataset defined in the flow field construction examples
example_dataset_name = "20250409_20X"

# Load the manifest and get the dataframe location for this dataset
feature_manifest_name = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED
feature_manifest = load_dataframe_manifest(feature_manifest_name)

dataset_location = get_dataframe_location_for_dataset(feature_manifest, example_dataset_name)

# Load only columns needed for flow field estimation
column_names = list(DYNAMICS_COLUMN_NAMES)
columns_to_load = [*METADATA_COLUMNS_TO_KEEP["grid"], *column_names]

df_raw = load_dataframe(dataset_location, delay=True)
df = df_raw[columns_to_load].compute()

# %%
# Filter to steady-state timepoints and a single flow condition
dataset_config = load_dataset_config(example_dataset_name)
dataframe_steady_state = filter_dataframe_to_steady_state(df, dataset_config)

# %% [markdown]
# ## Visualize binning + kernel convolution for displacements in a single bin (2D)
#
# This section illustrates the 2D Kramers-Moyal estimation for the `(r, rho)`
# coordinates:
# 1. **Binning**: trajectory positions are counted into 2D histogram bins
# 2. **Weighted histogram**: each position is weighted by the corresponding
#    r-displacement (forward difference), giving a numerator proportional to r-drift
# 3. **Kernel convolution**: a product kernel centred at each target bin
#    smooths the raw per-bin displacement estimates across neighbouring bins
# 4. **Single-bin highlight**: the panels zoom in on one representative bin to
#    make the weighting/smoothing step concrete

# %%
# --- choose the 2D features to illustrate: (r, rho) ---
feature_col_r = Column.DiffAEData.POLAR_RADIUS
feature_col_rho = Column.DiffAEData.PC3_FLIPPED
feature_cols = [feature_col_r, feature_col_rho]

# get 2D trajectories and displacements
traj_list, disp_list = get_traj_and_diff(dataframe_steady_state, feature_cols)

# build 2D bins from trajectory positions
bin_width_r = BIN_WIDTHS_DYNAMICS[feature_col_r]
bin_width_rho = BIN_WIDTHS_DYNAMICS[feature_col_rho]
positions_for_bins = np.concatenate([t[:-1] for t in traj_list], axis=0)  # shape (N, 2)
bins, centers = get_bins(
    bin_widths=(bin_width_r, bin_width_rho),
    data=positions_for_bins,
)
bin_centers_r = centers[0]
bin_centers_rho = centers[1]

# build per-dimension kernels
kernel_r = KramersMoyalKernel(
    name=KERNEL_NAMES_DYNAMICS[feature_col_r],
    bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[feature_col_r],
)
kernel_rho = KramersMoyalKernel(
    name=KERNEL_NAMES_DYNAMICS[feature_col_rho],
    bandwidth=KERNEL_BANDWIDTHS_DYNAMICS[feature_col_rho],
)
kernel_list = [kernel_r, kernel_rho]

# %%
# --- compute 2D histograms using km_computation helpers ---
# _get_km_powers(2) returns powers [[0,0],[1,0],[0,1],[2,0],[0,2]]
# rows: density, drift_r, drift_rho, diffusion_rr, diffusion_rho_rho
powers = _get_km_powers(ndim=2)
traj_list_, disp_list_, powers_ = _check_and_adjust_km_inputs(traj_list, disp_list, powers)

# weighted_hist shape: (5, n_bins_r, n_bins_rho)
weighted_hist = _get_weighted_histogram_for_convolution(traj_list_, disp_list_, bins, powers_)
disp_weighted_counts_r = weighted_hist[1]  # r-displacement-weighted counts (powers=[1,0])

# choose a representative target bin (bin closest to median position in each dim)
target_idx_r = int(np.argmin(np.abs(bin_centers_r - np.median(positions_for_bins[:, 0]))))
target_idx_rho = int(np.argmin(np.abs(bin_centers_rho - np.median(positions_for_bins[:, 1]))))
target_r = bin_centers_r[target_idx_r]
target_rho = bin_centers_rho[target_idx_rho]

# evaluate 2D product kernel weights centred at the target bin
offsets_r = bin_centers_r - target_r  # shape (n_bins_r,)
offsets_rho = bin_centers_rho - target_rho  # shape (n_bins_rho,)
offsets_grid = get_cartesian_product([offsets_r, offsets_rho])  # shape (n_bins_r, n_bins_rho, 2)
kernel_weights_2d = _evaluate_multivariate_product_kernel(
    offsets_grid, kernel_list
)  # (n_bins_r, n_bins_rho)

# kernel-smoothed r-drift at the target bin via get_kramers_moyal_coeffs
drift, _ = get_kramers_moyal_coeffs(traj_list, disp_list, bins, TIME_STEP_IN_HOURS, kernel_list)
kernel_drift_r_at_target = float(drift[target_idx_r, target_idx_rho, 0])

print(
    f"Target bin centre: r={target_r:.3f}, rho={target_rho:.3f}  |  "
    f"kernel-smoothed r-drift: {kernel_drift_r_at_target:.4f} [a.u./hr]"
)


# %%
# Helper for 2D pcolormesh panels
def _pcolormesh_2d(ax, data_2d, centers_r, centers_rho, cmap, label, vmin=None, vmax=None):
    """Plot a 2D heatmap with r on the x-axis and rho on the y-axis."""
    dr = centers_r[1] - centers_r[0]
    drho = centers_rho[1] - centers_rho[0]
    edges_r = np.append(centers_r - dr / 2, centers_r[-1] + dr / 2)
    edges_rho = np.append(centers_rho - drho / 2, centers_rho[-1] + drho / 2)
    pcm = ax.pcolormesh(
        edges_r,
        edges_rho,
        data_2d.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlabel(feature_col_r)
    ax.set_ylabel(feature_col_rho)
    return pcm


# --- zoom window: (factor x the kernel bandwidth around the target bin) ---
_zoom_factor = 3.5
_r_half = _zoom_factor * KERNEL_BANDWIDTHS_DYNAMICS[feature_col_r]
_rho_half = _zoom_factor * KERNEL_BANDWIDTHS_DYNAMICS[feature_col_rho]
_xlim = (target_r - _r_half, target_r + _r_half)
_ylim = (target_rho - _rho_half, target_rho + _rho_half)


def _add_target_bin_border(ax, color="magenta", linewidth=2.5, label="target bin"):
    """Draw a square border around the target bin on ax."""
    rect = mpatches.Rectangle(
        (target_r - bin_width_r / 2, target_rho - bin_width_rho / 2),
        bin_width_r,
        bin_width_rho,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label=label,
        zorder=5,
    )
    ax.add_patch(rect)


fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
fig.suptitle(
    f"Binning + kernel convolution for ({feature_col_r}, {feature_col_rho})  ·  dataset: {example_dataset_name}",
    fontsize=10,
)
# figsize: tuple[float, float] = (MAX_FIGURE_WIDTH, MAX_FIGURE_WIDTH // 4),

# panel 1 - r-displacement-weighted 2D histogram
ax = axes[0]
vmax2 = np.nanpercentile(np.abs(disp_weighted_counts_r), 99)
pcm = _pcolormesh_2d(
    ax,
    disp_weighted_counts_r,
    bin_centers_r,
    bin_centers_rho,
    cmap="RdBu_r",
    label=r"$\Sigma\,\Delta r$",
    vmin=-vmax2,
    vmax=vmax2,
)
_add_target_bin_border(ax)
ax.set_xlim(_xlim)
ax.set_ylim(_ylim)
fig.colorbar(pcm, ax=ax, label=r"$\Sigma\,\Delta r$")
ax.set_title("1. Weight by $r$-displacement")

# panel 2 - 2D kernel weights centred at target bin
ax = axes[1]
pcm = _pcolormesh_2d(
    ax,
    kernel_weights_2d / kernel_weights_2d.max(),
    bin_centers_r,
    bin_centers_rho,
    cmap="Purples",
    label="kernel weight",
)
_add_target_bin_border(ax)
ax.set_xlim(_xlim)
ax.set_ylim(_ylim)
fig.colorbar(pcm, ax=ax, label="normalised weight")
ax.set_title("2. Kernel centred at target bin")

# panel 3 - kernel-weighted r-displacement contributions
contrib_2d = kernel_weights_2d * disp_weighted_counts_r
ax = axes[2]
vmax4 = np.nanpercentile(np.abs(contrib_2d), 99)
pcm = _pcolormesh_2d(
    ax,
    contrib_2d,
    bin_centers_r,
    bin_centers_rho,
    cmap="RdBu_r",
    label=r"kernel $\times$ $\Delta r$",
    vmin=-vmax4,
    vmax=vmax4,
)
_add_target_bin_border(ax)
ax.set_xlim(_xlim)
ax.set_ylim(_ylim)
fig.colorbar(pcm, ax=ax, label=r"kernel $\times$ $\Delta r$")
ax.set_title("3. Kernel-weighted contributions")

# panel 4 - final r-drift field with target bin highlighted
ax = axes[3]
drift_r_2d = drift[..., 0]  # shape (n_bins_r, n_bins_rho)
pcm = _pcolormesh_2d(
    ax,
    drift_r_2d,
    bin_centers_r,
    bin_centers_rho,
    cmap=DRIFT_CONTOUR_COLORMAP,
    label=r"$r$-drift",
    vmin=DRIFT_CONTOUR_VMIN,
    vmax=DRIFT_CONTOUR_VMAX,
)
_add_target_bin_border(ax)
fig.colorbar(pcm, ax=ax, label=r"$r$-drift [hr$^{-1}$]")
ax.set_title("4. Final $r$-drift")
ax.set_xlim(_xlim)
ax.set_ylim(_ylim)

save_path = output_path / "binning_kernel_convolution_single_bin.pdf"
fig.savefig(save_path, bbox_inches="tight")
print(f"Saved figure to {save_path}")
plt.show()
# %%
