from endo_pipeline.cli import Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

KERNEL_BANDWIDTH = 0.175

BIN_WIDTH = 0.05

TICK_STEP_NUM = 15

NUM_INITS = 500  # number of initial points to sample for root solver

SHOW_SAMPLED_INITS = False


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    bw: float = KERNEL_BANDWIDTH,
    show_sampled_inits: bool = SHOW_SAMPLED_INITS,
) -> None:
    """
    Analyze and visualize DiffAE feature dynamics in polar coordinates.

    This workflow computes and visualizes the drift and diffusion coefficients
    in polar coordinates (angle and radius) for the grid-based crop features.
    The polar coordinates are computed from the first two principal components (PCs)
    of the DiffAE feature space as:
        - Angle: arctan2(PC2, PC1)
        - Radius: sqrt(PC1^2 + PC2^2)

    For each dataset in the specified collection, the workflow performs the following steps:
    1. Loads the grid-based crop feature dataframe and fits PCA to obtain the first two PCs
        and the corresponding polar coordinates.
    2. Splits the dataframe by flow conditions based on shear stress.
    3. For each flow condition:
        a. Plots the mean polar angle and radius over time for each position.
        b. Plots histogram heatmaps of polar angle and radius over time.
        c. Computes the Kramers-Moyal coefficients (drift and diffusion) for polar angle
            and radius using kernel density estimation (computes separately for each, i.e.,
            1D analysis in each of radius and angle).
        d. Plots the drift and diffusion coefficients along with fixed points identified
            from the drift function.
    """
    import logging
    import re

    import numpy as np
    from numdifftools import Jacobian
    from scipy.stats import gaussian_kde

    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        get_traj_and_diff,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.dynamics_utils.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
        sample_from_density,
    )
    from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
        plot_1d_diffusion,
        plot_1d_drift,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        plot_per_position_average,
        plot_principal_component_histogram,
    )
    from endo_pipeline.library.visualize.diffae_features.pplane import (
        STABILITY_COLOR_DICT,
        STABILITY_MARKER_DICT,
        build_phase_portrait_legend,
        find_fpt_type,
        get_fps,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

    logger = logging.getLogger(__name__)

    # notebook constant
    SPLIT_THETA_DATASETS = [
        "20250402_20X",
        "20250409_20X",
        "20250428_20X",
        "20250604_20X",
        "20250618_20X",
        "20250716_20X",
    ] + load_dataset_collection_config("perturbation").datasets

    DATASET_COLLECTION_NAME = "timelapse"

    POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]
    VARIABLE_NAMES = ["polar $\\theta$", "polar $r$"]

    BIN_LIMITS_THETA = (-np.pi, np.pi)
    BIN_LIMITS_RADIUS = (0, 2.75)

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first two PCs
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        dataset_names = get_datasets_in_collection(DATASET_COLLECTION_NAME, valid_dataset_options)
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        fig_savedir_summary = get_output_path(__file__, "summary_plots", dataset_name)
        fig_savedir_km = get_output_path(__file__, "kramers_moyal", dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            bin_limits = [BIN_LIMITS_THETA, BIN_LIMITS_RADIUS]

            bins, centers = get_bins(
                bin_widths=(BIN_WIDTH, BIN_WIDTH),
                bin_limits=bin_limits,
            )

            fig, ax = plot_per_position_average(
                df_,
                POLAR_COLUMN_NAMES,
            )
            save_plot_to_path(
                fig, fig_savedir_summary, f"{dataset_name_flow}_per_position_averages"
            )

            hist_arrays = []

            for i, column_name in enumerate(POLAR_COLUMN_NAMES):
                # plot histogram heatmap over time
                num_bins = len(bins[i]) - 1
                frame_min = df_[ColumnName.TIMEPOINT].min()
                frame_max = df_[ColumnName.TIMEPOINT].max()
                logger.debug("Frame min: [ %d ], Frame max: [ %d ]", frame_min, frame_max)
                num_frames = frame_max - frame_min + 1
                hist_array = np.zeros((num_bins, num_frames))

                for t, df_frame in df_.groupby(ColumnName.TIMEPOINT):
                    timepoint_idx = int(t - frame_min)
                    values = df_frame[column_name].values
                    hist = np.histogram(values, bins=bins[i], density=True)[0]
                    hist_array[:, timepoint_idx] = hist

                hist_arrays.append(hist_array)

            fig, ax = plot_principal_component_histogram(
                hist_arrays,
                bins,
                frame_range=(frame_min, frame_max),
                feature_names=VARIABLE_NAMES,
                bin_tick_step=TICK_STEP_NUM,
            )
            fig.suptitle(fig_title)
            save_plot_to_path(
                fig, fig_savedir_summary, f"{dataset_name_flow}_polar_histogram_heatmap"
            )

            # compute Kramers-Moyal coefficients

            # reset bin limits if shifting theta range
            # done to better capture dynamics near -pi to pi boundary
            is_low_shear_regime = shear_stress < 11.0 and shear_stress > 0.0
            shift_theta_range = is_low_shear_regime or (dataset_name in SPLIT_THETA_DATASETS)

            if shift_theta_range:
                df_[ColumnName.POLAR_ANGLE] = df_[ColumnName.POLAR_ANGLE].apply(
                    lambda x: x + 2 * np.pi if x < 0 else x
                )
                bin_limits = [(0, 2 * np.pi), BIN_LIMITS_RADIUS]

            bins, centers = get_bins(
                bin_widths=(BIN_WIDTH, BIN_WIDTH),
                bin_limits=bin_limits,
            )

            for i, column_name in enumerate(POLAR_COLUMN_NAMES):
                traj_list, d_traj_list = get_traj_and_diff(df_, [column_name])

                drift, diffusion = get_kramers_moyal(
                    traj_list,
                    d_traj_list,
                    bins=[bins[i]],
                    dt=5,
                    kernel_params={"kernel": "gaussian", "bandwidth": bw},
                )

                data_values = df_[column_name].values
                prob_density = gaussian_kde(data_values[np.newaxis, :], bw_method=bw)
                density_values = prob_density(centers[i])

                # shift theta back to [-pi, pi] if needed
                if column_name == ColumnName.POLAR_ANGLE and shift_theta_range:
                    # find where x > pi in centers and shift those values
                    idx_gt_pi = np.where(centers[i] > np.pi)[0]
                    centers[i][idx_gt_pi] = centers[i][idx_gt_pi] - 2 * np.pi

                    # sort by so that values of centers are in ascending order
                    idx_sorted = np.argsort(centers[i])
                    drift = drift[idx_sorted]
                    diffusion = diffusion[idx_sorted]
                    centers[i] = centers[i][idx_sorted]
                    density_values = density_values[idx_sorted]

                fig, ax = plot_1d_drift(
                    centers[i],
                    drift,
                    VARIABLE_NAMES[i],
                    density=density_values,
                )

                extrapolated_flow_field_dict = compute_extrapolated_vector_field(
                    drift, [centers[i]], method="linear", for_vtk_files=False
                )
                # get callable drift function and its Jacobian
                drift_function = get_callable_vector_field(
                    extrapolated_flow_field_dict, for_solve_ivp=False, method="linear"
                )
                drift_function_jacobian = Jacobian(drift_function)

                sampled_inits_for_root_solver = sample_from_density(
                    data_values, NUM_INITS, density=prob_density
                )

                if column_name == ColumnName.POLAR_ANGLE and shift_theta_range:
                    # shift sampled initial points > pi to be within [-pi, pi]
                    idx_points_gt_pi = np.where(sampled_inits_for_root_solver > np.pi)[0]
                    sampled_inits_for_root_solver[idx_points_gt_pi] = (
                        sampled_inits_for_root_solver[idx_points_gt_pi] - 2 * np.pi
                    )

                    # shift data values used for percentile check
                    idx_data_gt_pi = np.where(data_values > np.pi)[0]
                    data_values[idx_data_gt_pi] = data_values[idx_data_gt_pi] - 2 * np.pi

                if show_sampled_inits:
                    # plot sampled initial points for root solver - sanity check
                    ax.scatter(
                        sampled_inits_for_root_solver,
                        ax.get_ylim()[0] * np.ones_like(sampled_inits_for_root_solver),
                        s=1,
                        c="magenta",
                        marker="x",
                        alpha=0.3,
                        label="Sampled inits. for root solver",
                    )

                # pass into helper function to get fixed points
                fpts = get_fps(drift_function, sampled_inits_for_root_solver)
                stable_fpts = []
                fpt_stabilities = []
                for fpt in fpts:
                    # if outside the range of data, skip
                    if not (fpt[0] > data_values.min() and fpt[0] < data_values.max()):
                        logger.debug(
                            "Fixed point at [ %.2f ] is outside data range, skipping.",
                            fpt[0],
                        )
                        continue

                    # get stability and type of the fixed point
                    fpt_type = find_fpt_type(drift_function_jacobian(fpt))
                    fpt_stability = fpt_type.split(" ")[0].lower()
                    fpt_stabilities.append(fpt_stability)
                    # stability of the fixed point is the
                    # first word in the fpt_type string
                    # if verbose, print the point and its stability
                    logger.debug("[ %s ] at [ %.2f ]", fpt_type, fpt[0])
                    # plot the fixed point on the drift plot
                    ax.plot(
                        fpt[0],
                        0,
                        marker=STABILITY_MARKER_DICT[fpt_stability],
                        color=STABILITY_COLOR_DICT[fpt_stability],
                        markersize=8,
                    )
                    # if "Stable" or "stable" in the fpt_type, save the point
                    if re.search(r"stable", fpt_type, re.IGNORECASE) and not re.search(
                        r"unstable", fpt_type, re.IGNORECASE
                    ):
                        stable_fpts.append(fpt)

                # add legend for fixed point stabilities
                my_handles = build_phase_portrait_legend(
                    fpt_stabilities,
                    inits=None,
                    nullclines=False,
                )
                handles, _ = ax.get_legend_handles_labels()
                handles_new = handles + my_handles
                ax.legend(handles=handles_new, bbox_to_anchor=(1.05, 1.01), loc="upper left")
                ax.set_title(fig_title)
                save_plot_to_path(fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_drift")

                # make similar (but simpler) plot for diffusion coefficient
                fig, ax = plot_1d_diffusion(
                    centers[i],
                    diffusion,
                    VARIABLE_NAMES[i],
                    density=density_values,
                )
                ax.legend()
                ax.set_title(dataset_name)
                save_plot_to_path(
                    fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_diffusion"
                )


if __name__ == "main":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
