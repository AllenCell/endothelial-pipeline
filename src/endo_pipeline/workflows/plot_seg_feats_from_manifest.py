import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.configs.dataset_io import (
    ipython_cli_flexecute,
    parse_generate_dataset_name_user_input,
)
from endo_pipeline.io import configure_logging, get_output_path, load_dataframe
from endo_pipeline.library.analyze.live_data_manifest.lib_make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)
from endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
    hist_2D_of_feats,
    lineplot_of_feats,
    mark_parallel,
    mark_perpendicular,
)
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest

logger = logging.getLogger(__name__)


def plot_seg_manifest_data(
    seg_feats_df_subset: pd.DataFrame,
    dataset_name: str,
    position: int,
    out_dir: Path,
    show_plot: bool = False,
) -> None:
    """
    Creates and saves line plots and histograms of select segmentation
    features for a given dataset and position. Not all features are
    plotted as histograms.

    The features that are plotted are:
    - alignment
    - nematic order
    - eccentricity
    - aspect ratio
    - area
    - number of neighbors
    - centroid velocity magnitude
    - centroid velocity orientation
    - cell-nucleus distance
    - cell-nucleus orientation
    - number of nuclei (line plot only)
    - number of tracks (line plot only)
    """

    # choose which features to put on the y-axis
    # (we will put time on the x-axis)
    feats_to_plot_y = [
        "alignment_deg",
        "nematic_order",
        "eccentricity",
        "aspect_ratio",
        "area_um2",
        "num_neighbors",
        "centroid_velocity_magnitude",
        "centroid_velocity_orientation_deg",
        "cell_nuc_dist",
        "cell_nuc_orientation_deg",
        "cell_nuc_orientation_deg_rel_to_migration",
    ]
    feats_to_plot_y_lineplot_only = [
        "num_nuclei",
        "num_tracks",
    ]

    # get the plotting arguments for the features
    # (e.g. axis limits, axis titles, bin widths, etc.)
    feats_plot_args = get_seg_feat_plot_args()

    for feat in feats_to_plot_y + feats_to_plot_y_lineplot_only:
        filename_out = f"{dataset_name}_P{position}_{feat}.png"

        # plot alignment vs time as line plots
        out_subdir_lineplots = out_dir / "lineplots" / f"{feat}/{dataset_name}"
        out_subdir_lineplots.mkdir(parents=True, exist_ok=True)

        fig, ax = lineplot_of_feats(
            df_group=seg_feats_df_subset,
            x_column_name=feats_plot_args["time_hrs"]["column_name"],
            y_column_name=feats_plot_args[feat]["column_name"],
            x_label=feats_plot_args["time_hrs"]["label"],
            y_label=feats_plot_args[feat]["label"],
            y_lims=feats_plot_args[feat]["lims"],
            set_xticks=feats_plot_args["time_hrs"]["ticks"],
            set_yticks=feats_plot_args[feat]["ticks"],
            discrete_xticks=feats_plot_args["time_hrs"]["discrete_ticks"],
            discrete_yticks=feats_plot_args[feat]["discrete_ticks"],
            minor_ticks="xy",
        )
        fig.savefig(out_subdir_lineplots / filename_out, bbox_inches="tight")

        if not show_plot:
            plt.close(fig)

        if feat not in feats_to_plot_y_lineplot_only:
            # plot alignment vs time as histograms
            out_subdir_histplots = out_dir / "histplots" / f"{feat}/{dataset_name}"
            out_subdir_histplots.mkdir(parents=True, exist_ok=True)
            filename_out = f"{dataset_name}_P{position}_{feat}.png"

            fig, ax = hist_2D_of_feats(
                seg_feats_df_subset,
                x_column_name=feats_plot_args["time_hrs"]["column_name"],
                y_column_name=feats_plot_args[feat]["column_name"],
                x_label=feats_plot_args["time_hrs"]["label"],
                y_label=feats_plot_args[feat]["label"],
                x_lims=feats_plot_args["time_hrs"]["lims"],
                y_lims=feats_plot_args[feat]["lims"],
                set_xticks=feats_plot_args["time_hrs"]["ticks"],
                set_yticks=feats_plot_args[feat]["ticks"],
                discrete_xticks=feats_plot_args["time_hrs"]["discrete_ticks"],
                discrete_yticks=feats_plot_args[feat]["discrete_ticks"],
                minor_ticks="xy",
                bin_width=(
                    feats_plot_args["time_hrs"]["bin_width"],
                    feats_plot_args[feat]["bin_width"],
                ),
            )
            if "orientation" in feat:
                ax = mark_parallel(ax)
                ax = mark_perpendicular(ax)
            fig.savefig(out_subdir_histplots / filename_out, bbox_inches="tight")

            if not show_plot:
                plt.close(fig)


def process_dataset(dataset_name: str, out_dir: Path) -> None:
    """
    Loads the segmentation features manifest for a given dataset,
    calculates dynamic features, and generates and saves plots for
    each position.
    """

    # load the segmentation features table
    segprops_manifest = load_dataframe_manifest("live_merged_seg_features")
    segprops_location = get_dataframe_location_for_dataset(segprops_manifest, dataset_name)
    segprops_dataframe = load_dataframe(segprops_location)

    # get the FMS ID for the live merged segmentation features
    # and add it to the log
    logger.info(f"Dataset {dataset_name} FMS ID: {segprops_location.fmsid}")

    # apply the data filter
    segprops_dataframe = segprops_dataframe[segprops_dataframe["is_included"]]

    # iterate over each position in each dataset
    for (dataset_nm, pos), df_group in tqdm(
        segprops_dataframe.groupby(["dataset_name", "position"]),
        total=len(segprops_dataframe.groupby(["dataset_name", "position"])),
        desc=f"Plotting features: {dataset_name}",
        unit="position",
    ):
        # calculate the dynamics-dependent features
        df_group = calculate_derived_data_dynamics_dependent(df_group)

        # make some plots
        plot_seg_manifest_data(
            seg_feats_df_subset=df_group,
            dataset_name=dataset_nm,
            position=pos,
            out_dir=out_dir,
        )


def main(dataset_name: str | None = None, n_proc: int = 1, is_test: bool = False) -> None:

    dataset_name_list = parse_generate_dataset_name_user_input(dataset_name)
    print(f"Processing: {dataset_name_list}")

    out_dir = get_output_path(__file__)

    configure_logging(out_dir, logger, verbose=True)

    if n_proc > 1:
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            list(
                tqdm(
                    executor.map(
                        process_dataset, dataset_name_list, [out_dir] * len(dataset_name_list)
                    ),
                    total=len(dataset_name_list),
                    desc="Creating plots (MP)",
                    unit="dataset",
                )
            )

    else:
        for dataset in tqdm(
            dataset_name_list,
            total=len(dataset_name_list),
            desc="Creating plots (SP)",
            unit="dataset",
        ):
            # process dataset below will both load and plot the data
            process_dataset(dataset, out_dir)
            if is_test:
                break


if __name__ == "__main__":
    ipython_cli_flexecute(main)
