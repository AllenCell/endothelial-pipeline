from typing import Annotated

from cyclopts import Parameter

from endo_pipeline.cli import CropPattern, Datasets
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)


def main(
    datasets: Datasets | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    crop_pattern: CropPattern = "grid",
    global_axes_limits: bool = False,
    just_steady_state: Annotated[bool, Parameter(negative="--include-transient")] = True,
) -> None:
    """
    Analyze and visualize DiffAE feature dynamics in polar coordinates.

    This workflow computes and visualizes the dynamics of DiffAE features
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

    Parameters
    ----------
    datasets
        The datasets to process. If None, uses the default dataset collection.
    model_manifest_name
        The name of the model manifest to use.
    run_name
        The name of the model run to use.
    crop_pattern
        The crop pattern to get features for, either "grid" or "tracked".
    global_axes_limits
        Whether to use global axes limits for the per-position average plots.
    rescale_theta
        Whether to rescale theta values to [0, pi] range with period pi.
    """

    import numpy as np

    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.numerics.binning import get_bins
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_label_for_column,
        plot_component_histograms_over_time,
        plot_per_position_average_over_time,
    )
    from endo_pipeline.manifests import (
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.polar_coords import (
        BIN_LIMITS_POLAR,
        BIN_LIMITS_THETA_RESCALED,
        BIN_WIDTHS_POLAR,
        DEFAULT_DATASET_COLLECTION_POLAR_VIS,
        POLAR_COLUMN_NAMES,
        RESCALE_THETA,
        TICK_STEP_NUM,
    )

    # get feature column names and labels (for plots)
    column_names = [*POLAR_COLUMN_NAMES, "pc_3"]
    variable_names = [get_label_for_column(col) for col in column_names]

    # get dataframe manifest for grid-based crop features
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern=crop_pattern
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

    # only need first two PCs
    pca = fit_pca(dataframe_manifest_name=dataframe_manifest_name, num_pcs=3)

    # Default list of datasets if not provided, only include datasets available in
    # the provided dataframe manifest
    valid_dataset_options = list(dataframe_manifest.locations.keys())
    if datasets is None:
        dataset_names = get_datasets_in_collection(
            DEFAULT_DATASET_COLLECTION_POLAR_VIS, valid_dataset_options
        )
    else:
        dataset_names = [name for name in datasets if name in valid_dataset_options]

    # compute bins for polar coordinates
    bin_limits = [*BIN_LIMITS_POLAR, (-3, 3)]
    idx_theta = column_names.index(ColumnName.POLAR_ANGLE.value)
    if RESCALE_THETA:
        bin_limits[idx_theta] = BIN_LIMITS_THETA_RESCALED

    bins, _ = get_bins(
        bin_widths=(BIN_WIDTHS_POLAR[0], BIN_WIDTHS_POLAR[1], 0.05),
        bin_limits=bin_limits,
    )

    # set output director based on whether including transient data
    file_subdir = "just_steady_state" if just_steady_state else "includes_transient_data"

    # loop over datasets in collection
    # plot summary plots
    # compute drift and diffusion coefficients in polar coordinates
    for dataset_name in dataset_names:
        fig_savedir = get_output_path(__file__, file_subdir, dataset_name)
        dataset_config = load_dataset_config(dataset_name)

        df = get_dataframe_for_dynamics_workflows(
            dataset_name,
            dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=not just_steady_state,
            compute_polar=True,
            rescale_theta=RESCALE_THETA,
        )

        df_by_flow, shear_stress_list = split_dataset_by_flow(
            df,
            dataset_config,
        )

        for df_, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            # for datasets with theta distribution similar to MIN shear stress,
            # shift polar angle range from (-pi, pi) to (0, 2pi) to avoid
            # numerical errors that come from angle wrapping around at -pi/pi boundary

            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            fig_title = f"{dataset_name} ({shear_stress} dym/cm$^2$)"

            fig, ax = plot_per_position_average_over_time(
                df_,
                column_names=column_names,
                column_labels=variable_names,
                polar_angle_range=bin_limits[idx_theta],
            )
            if global_axes_limits:
                for i, ax_ in enumerate(ax):
                    ax_.set_ylim(bin_limits[i])

            fig.suptitle(fig_title, y=0.91)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_per_position_averages")

            hist_arrays = []

            for i, column_name in enumerate(column_names):
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

                hist_arrays.append(hist_array)

            fig, ax = plot_component_histograms_over_time(
                hist_arrays,
                bins,
                frame_range=(frame_min, frame_max),
                feature_names=variable_names,
                time_tick_step=50,
                bin_tick_num=TICK_STEP_NUM,
            )
            fig.suptitle(fig_title, y=0.91)
            save_plot_to_path(fig, fig_savedir, f"{dataset_name_flow}_histogram_heatmap")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
