# %%
import logging
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from endo_pipeline.cli import DEMO_MODE
from endo_pipeline.configs import (
    get_datasets_in_collection,
    get_shear_stress_label_for_dataset,
    load_dataset_config,
)
from endo_pipeline.io import get_output_path, join_sorted_strings, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_shear_stress
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.sde_interpretation import (
    extrapolate_noise_amplitude_tensor,
    get_reshaped_noise_amplitude_and_grid,
)
from endo_pipeline.library.visualize.columns import get_label_for_column
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_DYNAMICS,
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
)
from endo_pipeline.settings.manifest_names import NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES
from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

# %%
logger = logging.getLogger(__name__)

patch_type = "grid_based"
output_path = get_output_path(__file__, patch_type)

dataset_names = get_datasets_in_collection("shear_stress", "perturbation")

if DEMO_MODE:
    logger.warning("DEMO MODE - Limiting to one dataset")
    dataset_names = dataset_names[:1]

# Use all three dynamics columns, matching the compute-noise-amplitude output
column_names = list(DYNAMICS_COLUMN_NAMES)
column_labels = [get_label_for_column(column) for column in column_names]
mesh_column_names = [ColumnTemplate.MESH_GRID % column for column in column_names]
num_dimensions = len(column_names)

# Unique (i, j) entries and their panel titles/column names, labeled with i <= j
entry_pairs = list(combinations_with_replacement(range(num_dimensions), 2))
entry_labels = [f"{Unicode.SIGMA}({column_labels[i]}, {column_labels[j]})" for i, j in entry_pairs]
entry_column_names = [
    ColumnTemplate.NOISE_AMPLITUDE % (column_names[i], column_names[j]) for i, j in entry_pairs
]
# sigma is the lower-triangular Cholesky factor, so the non-trivial entry
# for each (i, j) pair with i < j lives at tensor index (j, i), not (i, j).
tensor_index_pairs = [(j, i) for i, j in entry_pairs]
required_columns = [
    *mesh_column_names,
    *entry_column_names,
    Column.DATASET,
    Column.SHEAR_STRESS,
]

bin_widths = [BIN_WIDTHS_DYNAMICS[column] for column in column_names]
bounds_for_plots = [BIN_LIMITS_DYNAMICS[column] for column in column_names]

# Load precomputed noise amplitude tensor for selected columns
name_suffix = join_sorted_strings(column_names)
noise_amplitude_manifest_name = (
    f"{NOISE_AMPLITUDE_MANIFEST_NAME_PREFIXES[patch_type]}_{name_suffix}"
)
noise_amplitude_manifest = load_dataframe_manifest(noise_amplitude_manifest_name)

# (axes to plot in 2D, axis being sliced over) for each of the three slice stacks
plot_axes_indices = [(0, 1), (0, 2), (1, 2)]
slice_axis_indices = [2, 1, 0]

# %%
r_inflections = []
for dataset_name in dataset_names:
    if dataset_name not in noise_amplitude_manifest.locations:
        logger.warning(
            "Dataset '%s' not found in manifest '%s'. Skipping.",
            dataset_name,
            noise_amplitude_manifest_name,
        )
        continue

    dataset_config = load_dataset_config(dataset_name)

    noise_amplitude_dataframe_location = get_dataframe_location_for_dataset(
        noise_amplitude_manifest, dataset_name
    )
    noise_amplitude_dataframe = load_dataframe(noise_amplitude_dataframe_location, delay=False)
    check_required_columns_in_dataframe(noise_amplitude_dataframe, required_columns)

    for flow_condition in dataset_config.flow_conditions:
        shear_stress = flow_condition.shear_stress
        dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
        fig_title = get_shear_stress_label_for_dataset(dataset_config, flow_condition)

        noise_amplitude_for_flow_condition = filter_dataframe_by_shear_stress(
            noise_amplitude_dataframe, shear_stress
        )
        if noise_amplitude_for_flow_condition.empty:
            logger.warning(
                "No noise amplitude tensor found for [ %s ]. Skipping.", dataset_name_flow
            )
            continue

        sigma, grid_points_1d = get_reshaped_noise_amplitude_and_grid(
            noise_amplitude_for_flow_condition, column_names
        )
        # fill NaN bins (too few samples) before plotting, so sparse
        # per-bin estimates don't leave slices entirely blank
        sigma = extrapolate_noise_amplitude_tensor(sigma, grid_points_1d)
        grid = np.meshgrid(*grid_points_1d, indexing="ij")

        # plot sigma_11 as a function of r for a center slice of rho,
        # multiple slices of theta
        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("r")
        ax.set_ylabel("sigma_11")
        center_slice_index = grid[2].shape[2] // 2
        num_theta_slices = grid[0].shape[1] // 10
        r_values = grid[1][0, :, 0]
        mean_sigma_11 = np.mean(sigma[:, :, center_slice_index, 0, 0], axis=0)

        # inflection point of d(sigma_11)/dr where mean sigma_11(r) starts rising rapidly with
        # decreasing r: the local minimum of d^3 sigma_11/dr^3 in the mean profile, restricted to
        # r < 1.75 to avoid spurious minima in the flat large-r tail
        d_mean_sigma_11_dr = np.gradient(mean_sigma_11, r_values)
        d2_mean_sigma_11_dr2 = np.gradient(d_mean_sigma_11_dr, r_values)
        d3_mean_sigma_11_dr3 = np.gradient(d2_mean_sigma_11_dr2, r_values)
        r_search_mask = r_values < 1.75
        inflection_index = np.argmin(np.where(r_search_mask, d3_mean_sigma_11_dr3, np.inf))
        inflection_r = r_values[inflection_index]
        r_inflections.append(inflection_r)
        logger.info(
            "[ %s ] d(sigma_11)/dr inflection point (local min) at r = %.3f",
            dataset_name_flow,
            inflection_r,
        )

        for theta_index in range(0, grid[0].shape[1], num_theta_slices):
            theta_value = grid[0][theta_index, 0, 0]
            sigma_11_value = sigma[theta_index, :, center_slice_index, 0, 0]
            ax.plot(
                r_values,
                sigma_11_value,
                label=f"theta={theta_value:.3f}",
                linewidth=1.0,
                alpha=0.5,
            )
        ax.plot(
            r_values,
            mean_sigma_11,
            label="mean",
            linestyle="--",
            color="black",
        )
        ax.axvline(
            inflection_r, label=f"inflection r={inflection_r:.3f}", linestyle=":", color="gray"
        )
        ax.legend()
        save_plot_to_path(fig, output_path, f"{dataset_name_flow}_sigma_11_vs_r.png")

        # fit sum of exponentials to the mean sigma_11 as a function of r
        def sum_of_exponentials(r, *params):
            n = len(params) // 2
            result = np.zeros_like(r)
            for i in range(n):
                amplitude = params[2 * i]
                decay = params[2 * i + 1]
                result += amplitude * np.exp(decay * r)
            return result

        try:
            param_fit = curve_fit(
                sum_of_exponentials, r_values, mean_sigma_11, p0=[1.0, 1.0, 1.0, 1.0]
            )[0]
            # plot the fitted curve
            fit_label = "+".join(
                [
                    f"{param_fit[2*i]:.3f}*e^({param_fit[2*i+1]:.3f}*r)"
                    for i in range(len(param_fit) // 2)
                ]
            )
            fitted_sigma_11 = sum_of_exponentials(r_values, *param_fit)
            fig, ax = plt.subplots()
            ax.set_title(f"{fig_title} - Fitted")
            ax.set_xlabel("r")
            ax.set_ylabel("sigma_11 (mean)")
            ax.plot(r_values, mean_sigma_11, label="mean", linestyle="--", color="black")
            ax.plot(r_values, fitted_sigma_11, label=fit_label, linestyle="-", color="red")
            ax.axvline(
                inflection_r, label=f"inflection r={inflection_r:.3f}", linestyle=":", color="gray"
            )
            ax.legend()
            save_plot_to_path(fig, output_path, f"{dataset_name_flow}_sigma_11_vs_r_fit.png")
        except Exception as e:
            print(f"Failed to fit sum of exponentials: {e}")

        # plot the third partial derivative of mean sigma_11 w.r.t. r, used above to find the
        # inflection point of d(sigma_11)/dr
        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("r")
        ax.set_ylabel("d3(sigma_11)/dr3 (mean)")
        ax.plot(r_values, d3_mean_sigma_11_dr3, color="black")
        ax.axvline(
            inflection_r, label=f"inflection r={inflection_r:.3f}", linestyle=":", color="gray"
        )
        ax.legend()
        save_plot_to_path(fig, output_path, f"{dataset_name_flow}_d3sigma_11_dr3_vs_r.png")

        # partial derivative of sigma_11 w.r.t. r, via finite differences on the grid
        sigma_11 = sigma[..., 0, 0]
        axis, derivative_name = 1, "r"
        derivative = np.gradient(sigma_11, grid_points_1d[axis], axis=axis)
        derivative_label = f"d(sigma_11)/d{derivative_name}"

        # contour plot of the derivative over the (theta, r) plane, center rho slice
        fig, ax = plt.subplots()
        ax.set_title(f"{fig_title}\n{derivative_label}")
        ax.set_xlabel("theta")
        ax.set_ylabel("r")
        derivative_slice = derivative[:, :, center_slice_index]
        # center the diverging colormap at zero regardless of the data's sign skew
        abs_max = np.nanmax(np.abs(derivative_slice))
        contour = ax.contourf(
            grid[0][:, :, center_slice_index],
            grid[1][:, :, center_slice_index],
            derivative_slice,
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
        )
        fig.colorbar(contour, ax=ax, label=derivative_label)
        save_plot_to_path(
            fig, output_path, f"{dataset_name_flow}_dsigma_11_d{derivative_name}_contour.png"
        )

        # 1d plot of the derivative as a function of r, multiple slices of theta, center rho slice
        fig, ax = plt.subplots()
        ax.set_title(fig_title)
        ax.set_xlabel("r")
        ax.set_ylabel(derivative_label)
        mean_derivative_vs_r = np.mean(derivative[:, :, center_slice_index], axis=0)
        for theta_index in range(0, grid[0].shape[1], num_theta_slices):
            theta_value = grid[0][theta_index, 0, 0]
            ax.plot(
                r_values,
                derivative[theta_index, :, center_slice_index],
                label=f"theta={theta_value:.3f}",
                linewidth=1.0,
                alpha=0.5,
            )
        ax.plot(
            r_values,
            mean_derivative_vs_r,
            label="mean",
            linestyle="--",
            color="black",
        )
        ax.axvline(
            inflection_r,
            label=f"inflection r={inflection_r:.3f}",
            linestyle=":",
            color="gray",
        )
        ax.legend()
        save_plot_to_path(
            fig, output_path, f"{dataset_name_flow}_dsigma_11_d{derivative_name}_vs_r.png"
        )

print(
    "Mean r_inflection over %d processed datasets: %.3f", len(r_inflections), np.mean(r_inflections)
)
# %%
