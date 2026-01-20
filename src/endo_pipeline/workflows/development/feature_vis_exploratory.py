from endo_pipeline.cli import Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> None:
    import logging
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    from numdifftools import Jacobian

    from endo_pipeline.configs import (
        get_datasets_in_collection,
        load_dataset_collection_config,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.dynamics_utils.data_driven_flow_field import (
        compute_extrapolated_vector_field,
        get_callable_vector_field,
        sample_from_density,
    )
    from endo_pipeline.library.analyze.kramersmoyal import get_kramers_moyal
    from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
        get_smallest_angle_difference,
    )
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.dynamics_viz import (
        plot_1d_diffusion,
        plot_1d_drift,
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

    KERNEL_BANDWIDTH = 0.175  # bandwidth for kernel density estimation in KM calculation

    POLAR_COLUMN_NAMES = [ColumnName.POLAR_ANGLE, ColumnName.POLAR_RADIUS]

    BIN_WIDTH = 0.075

    TICK_STEP_NUM = [15, 5]

    BIN_LIMITS_THETA = (-np.pi, np.pi)
    BIN_LIMITS_RADIUS = (0, 2.75)

    NUM_INITS = 250  # number of initial points to sample for root solver

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

            is_low_shear_regime = shear_stress < 11.0 and shear_stress > 0.0

            if dataset_name in SPLIT_THETA_DATASETS or is_low_shear_regime:
                df_[ColumnName.POLAR_ANGLE] = df_[ColumnName.POLAR_ANGLE].apply(
                    lambda x: x + 2 * np.pi if x < 0 else x
                )
                bin_limits = [(0, 2 * np.pi), BIN_LIMITS_RADIUS]

            bins, centers = get_bins(
                bin_widths=(BIN_WIDTH, BIN_WIDTH),
                bin_limits=bin_limits,
            )

            for i, column_name in enumerate(POLAR_COLUMN_NAMES):
                # plot per position mean
                fig, ax = plt.subplots()

                for pos, df_pos in df_.groupby(ColumnName.POSITION):
                    df_pos_ = df_pos.sort_values(by=ColumnName.TIMEPOINT)
                    mean_over_crops = df_pos_.groupby(ColumnName.TIMEPOINT)[column_name].mean()
                    timepoints = df_pos_[ColumnName.TIMEPOINT].unique()
                    ax.plot(timepoints, mean_over_crops, label=pos)

                ax.legend()
                ax.set_xlabel("frame number")
                ax.set_ylabel(column_name)
                ax.set_title(fig_title)
                save_plot_to_path(
                    fig, fig_savedir_summary, f"{dataset_name_flow}_{column_name}_average"
                )

                # plot histogram heatmap over time
                num_bins = len(bins[i]) - 1
                frame_min = df_[ColumnName.TIMEPOINT].min()
                frame_max = df_[ColumnName.TIMEPOINT].max()
                num_frames = frame_max - frame_min + 1
                hist_array = np.zeros((num_bins, num_frames))

                for t, df_frame in df_.groupby(ColumnName.TIMEPOINT):
                    timepoint_idx = int(t - frame_min)
                    values = df_frame[column_name].values
                    hist = np.histogram(values, bins=bins[i], density=True)[0]
                    hist_array[:, timepoint_idx] = hist

                fig, ax = plt.subplots()
                ax.imshow(
                    hist_array,
                    aspect="auto",
                    cmap="inferno",
                    interpolation="nearest",
                    origin="lower",
                )

                ax.set_ylabel(column_name)
                ax.set_xlabel("frame_number")
                ax.set_xticks(np.arange(frame_min, frame_max + 1, step=100))
                ax.set_xticklabels(np.arange(frame_min, frame_max + 1, step=100))
                ax.set_yticks(np.arange(0, num_bins + 1, step=TICK_STEP_NUM[i]))
                ax.set_yticklabels(np.round(bins[i], 2)[:: TICK_STEP_NUM[i]])
                ax.set_title(fig_title)
                save_plot_to_path(
                    fig, fig_savedir_summary, f"{dataset_name_flow}_{column_name}_histogram_heatmap"
                )

            # compute Kramers-Moyal coefficients
            for i, column_name in enumerate(POLAR_COLUMN_NAMES):
                traj_list = []
                d_traj_list = []
                for crop_index, df_crop in df_.groupby(ColumnName.CROP_INDEX):
                    df_crop_ = df_crop.sort_values(by=ColumnName.TIMEPOINT)

                    # add column giving difference in timepoint between consecutive dataframe rows
                    # convert NaN to 0 -- occurs at end of trajectory
                    df_crop_[f"{ColumnName.TIMEPOINT}_diff"] = (
                        df_crop_[ColumnName.TIMEPOINT].diff().shift(-1).fillna(0)
                    )
                    # compute one-step differences in polar angle and radius
                    if column_name == ColumnName.POLAR_ANGLE:
                        angle_diffs = get_smallest_angle_difference(
                            df_crop_[column_name].values[1:],
                            df_crop_[column_name].values[:-1],
                            units="rad",
                        )
                        df_crop_[f"{column_name}_diff"] = np.concatenate(
                            (
                                angle_diffs,
                                np.array([np.nan]),
                            )  # no valid difference at end of trajectory
                        )
                    else:
                        df_crop_[f"{column_name}_diff"] = df_crop_[column_name].diff().shift(-1)

                    # trajectory values to keep -- only keep steps where time difference is 1
                    # and also the last point in the trajectory (which has time difference 0)
                    traj_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] <= 1

                    # for the gradient, only keep steps where time difference is exactly 1
                    # i.e., no valid difference at the end of the trajectory (only forward differences)
                    gradient_mask = df_crop_[f"{ColumnName.TIMEPOINT}_diff"] == 1

                    traj_list.append(df_crop_[traj_mask][column_name].values)
                    d_traj_list.append(df_crop_[gradient_mask][f"{column_name}_diff"].values)

                drift, diffusion = get_kramers_moyal(
                    traj_list,
                    d_traj_list,
                    bins=[bins[i]],
                    dt=5,
                    kernel_params={"kernel": "gaussian", "bandwidth": KERNEL_BANDWIDTH},
                )

                data_values = df_[column_name].values

                variable_name = "\\theta" if column_name == ColumnName.POLAR_ANGLE else "r"
                fig, ax = plot_1d_drift(
                    centers[i],
                    drift,
                    variable_name,
                    data_for_density=data_values,
                    density_kernel_bandwidth=KERNEL_BANDWIDTH,
                )

                extrapolated_flow_field_dict = compute_extrapolated_vector_field(
                    drift, centers[i], method="linear", for_vtk_files=False
                )
                # get callable drift function and its Jacobian
                drift_function = get_callable_vector_field(
                    extrapolated_flow_field_dict, for_solve_ivp=False, method="linear"
                )
                drift_function_jacobian = Jacobian(drift_function)

                sampled_inits_for_root_solver = sample_from_density(data_values, NUM_INITS)
                ax.scatter(
                    np.zeros_like(sampled_inits_for_root_solver),
                    sampled_inits_for_root_solver,
                    s=1,
                    c="magenta",
                    label="Sampled inits. for root solver",
                )

                # pass into helper function to get fixed points
                fpts = get_fps(drift_function, sampled_inits_for_root_solver)
                stable_fpts = []
                fpt_stabilities = []
                for fpt in fpts:
                    # get stability and type of the fixed point
                    fpt_type = find_fpt_type(drift_function_jacobian(fpt))
                    fpt_stability = fpt_type.split(" ")[0].lower()
                    fpt_stabilities.append(fpt_stability)
                    # stability of the fixed point is the
                    # first word in the fpt_type string
                    # if verbose, print the point and its stability
                    logger.debug(
                        "[ %s ] at [ (%.2f, %.2f, %.2f) ]", fpt_type, fpt[0], fpt[1], fpt[2]
                    )
                    # plot the fixed point on the drift plot
                    ax.plot(
                        fpt[0],
                        fpt[1],
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
                ax.legend(handles=handles_new, bbox_to_anchor=(1.02, 1.01), loc="upper left")
                ax.set_title(fig_title)
                save_plot_to_path(fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_drift")

                fig, ax = plot_1d_diffusion(
                    centers[i],
                    diffusion,
                    variable_name,
                    data_for_density=df_[column_name].values,
                    density_kernel_bandwidth=KERNEL_BANDWIDTH,
                )
                ax.legend()
                ax.set_title(dataset_name)
                save_plot_to_path(
                    fig, fig_savedir_km, f"{dataset_name_flow}_{column_name}_diffusion"
                )


if __name__ == "main":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
