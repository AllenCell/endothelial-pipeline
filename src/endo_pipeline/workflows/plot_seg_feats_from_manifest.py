import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.endo_pipeline.configs.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_segmentation_features_manifest,
    ipython_cli_flexecute,
    save_git_versioning_info,
)
from src.endo_pipeline.io import configure_logging, get_output_path
from src.endo_pipeline.library.visualize.seg_features.general_standard_plots import (
    get_seg_feat_plot_args,
    hist_2D_per_dataset,
    lineplot_per_dataset,
    mark_parallel,
    mark_perpendicular,
)
from src.endo_pipeline.workflows.make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)


def plot_seg_manifest_data(
    # seg_feats_df_subset: pd.DataFrame,
    big_table_subset: pd.DataFrame,
    dataset_name: str,
    # position: int,
    out_dir: Path,
    show_plot: bool = False,
) -> None:

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
    ]
    feats_to_plot_y_lineplot_only = [
        "num_nuclei",
        "num_tracks",
    ]

    feats_plot_args = get_seg_feat_plot_args()

    for feat in feats_to_plot_y + feats_to_plot_y_lineplot_only:
        y_key = feat
        out_subdir_plots = out_dir / "lineplots" / f"{y_key}/{dataset_name}"
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        fig, ax = lineplot_per_dataset(
            big_table_subset,
            x_column_name=feats_plot_args["time_hrs"]["column_name"],
            y_column_name=feats_plot_args[feat]["column_name"],
            x_label=feats_plot_args["time_hrs"]["label"],
            y_label=feats_plot_args[feat]["label"],
            y_lims=feats_plot_args[feat]["lims"],
        )
        fig.savefig(out_subdir_plots / feats_plot_args[feat]["filename_out"], bbox_inches="tight")

        if not show_plot:
            plt.close(fig)

    # plot alignment vs time as a histogram instead of violinplot
    for feat in feats_to_plot_y:
        out_subdir_plots = out_dir / "histplots" / f"{y_key}/{dataset_name}"
        out_subdir_plots.mkdir(parents=True, exist_ok=True)

        fig, ax = hist_2D_per_dataset(
            big_table_subset,
            x_column_name=feats_plot_args["time_hrs"]["column_name"],
            y_column_name=feats_plot_args[feat]["column_name"],
            x_label=feats_plot_args["time_hrs"]["label"],
            y_label=feats_plot_args[feat]["label"],
            x_lims=feats_plot_args["time_hrs"]["lims"],
            y_lims=feats_plot_args[feat]["lims"],
            bin_width=(feats_plot_args["time_hrs"]["lims"], feats_plot_args[feat]["bin_width"]),
        )
        if "orientation" in y_key:
            ax = mark_parallel(ax)
            ax = mark_perpendicular(ax)

        fig.savefig(out_subdir_plots / feats_plot_args[feat]["filename_out"], bbox_inches="tight")

        if not show_plot:
            plt.close(fig)


def process_dataset(dataset_name: str, out_dir: Path) -> None:
    # load the segmentation features table
    segprops_manifest = get_segmentation_features_manifest([dataset_name])

    # apply the data filter
    segprops_manifest = segprops_manifest[~segprops_manifest["filter_global"]]

    # make basic plots for each dataset
    for (dataset_nm, pos), df_group in tqdm(
        segprops_manifest.groupby(["dataset_name", "position"]),
        total=len(segprops_manifest.groupby(["dataset_name", "position"])),
        desc=f"Plotting features: {dataset_name}",
        unit="position",
    ):
        # calculate the dynamics-dependent features
        df_group = calculate_derived_data_dynamics_dependent(df_group)

        # make some plots
        plot_seg_manifest_data(
            big_table_subset=df_group,
            dataset_name=dataset_nm,
            # position=pos,
            out_dir=out_dir,
        )


def main(dataset_name: str | None = None, n_proc: int = 1) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
    dataset_name_list = [dataset_name_list[0]]
    print(f"Processing: {dataset_name_list}")

    out_dir = get_output_path(Path(__file__).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # save git versioning info
    save_git_versioning_info(out_dir, Path(__file__).stem, verbose=False)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ipython_cli_flexecute(main)
