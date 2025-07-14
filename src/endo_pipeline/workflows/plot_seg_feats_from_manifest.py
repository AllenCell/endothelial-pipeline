import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.endo_pipeline.configs.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_live_segmentation_features_manifest,
    ipython_cli_flexecute,
    save_git_versioning_info,
)
from src.endo_pipeline.io import configure_logging, get_output_path
from src.endo_pipeline.workflows.make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)

logger = logging.getLogger(__name__)

# set the plot shape to the golden ratio
AX_HEIGHT = 3
AX_WIDTH = AX_HEIGHT * (1 + 5 ** (1 / 2)) / 2


def lineplot_per_position(
    df_group: pd.DataFrame,
    x_key: str,
    y_key: str,
    filepath_out: str | Path,
    x_label: str | None = None,
    y_label: str | None = None,
    x_lims: tuple = (None, None),
    y_lims: tuple = (None, None),
    show_plot: bool = False,
) -> None:
    """
    This function will save a standardized lineplot from the dataframe df_group.
    x_key and y_key are the column names that you want to plot along the x-axis
    and y-axis, respectively.
    df_group is expected to contain a single dataset and a single position.

    Parameters
    ----------
    df_group : pd.DataFrame
        The dataframe containing the data to plot.
    x_key : str
        The column name for the x-axis data.
    y_key : str
        The column name for the y-axis data.
    filepath_out : str | Path
        The file path where the plot will be saved.
    x_label : str | None, optional
        The label for the x-axis, will use x_key as the default.
    y_label : str | None, optional
        The label for the y-axis, will use y_key as the default.
    x_lims : tuple, optional
        The limits for the x-axis as a (min, max) tuple.
    y_lims : tuple, optional
        The limits for the y-axis as a (min, max) tuple.
    show_plot : bool, optional
        Whether to show the plot after saving it. Default is False.
    """

    num_positions = df_group["position"].nunique()
    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]
    assert (
        len(df_group["position"].unique()) == 1
    ), f'Only a single position allowed in df_group, position found: {df_group["position"].unique()}'
    position = df_group["position"].unique()[0]

    fig, ax = plt.subplots(nrows=num_positions, figsize=(AX_WIDTH, AX_HEIGHT * num_positions))
    ax.set_title(f"{dataset_name} P{position}")
    sns.lineplot(data=df_group, x=x_key, y=y_key, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    plt.tight_layout()
    fig.savefig(filepath_out, bbox_inches="tight")

    if not show_plot:
        plt.close(fig)
    return


def plot_seg_manifest_data(
    big_table_subset: pd.DataFrame, dataset_name: str, position: int, out_dir: Path
) -> None:
    vel_mag_mean = big_table_subset["centroid_velocity_magnitude"].mean()
    vel_mag_std = big_table_subset["centroid_velocity_magnitude"].std()
    # things_to_plot are tuples of (x_key, y_key, x_label, y_label, y_lim, filename_out)
    things_to_plot = [
        (
            "time_hours",
            "alignment_deg_rel_to_flow",
            "Time (hours)",
            "Alignment (deg)",
            (0, 90),
            f"{dataset_name}_P{position}_alignments.png",
        ),
        (
            "time_hours",
            "eccentricity",
            "Time (hours)",
            "Eccentricity",
            (0, 1),
            f"{dataset_name}_P{position}_eccentricities.png",
        ),
        (
            "time_hours",
            "nematic_order",
            "Time (hours)",
            "Nematic Order",
            (-1, 1),
            f"{dataset_name}_P{position}_nematic_order.png",
        ),
        (
            "time_hours",
            "aspect_ratio",
            "Time (hours)",
            "Aspect Ratio",
            (0, None),
            f"{dataset_name}_P{position}_aspect_ratio.png",
        ),
        (
            "time_hours",
            "area",
            "Time (hours)",
            "Area (px**2)",
            (0, None),
            f"{dataset_name}_P{position}_region_areas.png",
        ),
        (
            "time_hours",
            "number_of_neighbors",
            "Time (hours)",
            "Number of Neighbors",
            (0, None),
            f"{dataset_name}_P{position}_num_neighbors.png",
        ),
        (
            "time_hours",
            "num_tracks_at_T",
            "Time (hours)",
            "Number of Cell Tracks",
            (0, None),
            f"{dataset_name}_P{position}_num_tracks.png",
        ),
        (
            "time_hours",
            "centroid_velocity_magnitude",
            "Time (hours)",
            "Centroid Velocity Magnitude (um/min)",
            (0, vel_mag_mean + 2 * vel_mag_std),
            f"{dataset_name}_P{position}_centroid_velocity_magnitudes.png",
        ),
        (
            "time_hours",
            "total_nuclei_count_at_T",
            "Time (hours)",
            "Number of Predicted Nuclei",
            (0, None),
            f"{dataset_name}_P{position}_num_nuclei.png",
        ),
    ]
    for x_key, y_key, x_label, y_label, y_lims, filename_out in things_to_plot:
        out_subdir_plots = out_dir / f"{y_key}/{dataset_name}"
        out_subdir_plots.mkdir(parents=True, exist_ok=True)
        lineplot_per_position(
            big_table_subset,
            x_key=x_key,
            y_key=y_key,
            filepath_out=out_subdir_plots / filename_out,
            x_label=x_label,
            y_label=y_label,
            y_lims=y_lims,
        )

    t_range = range(0, 1000, 36)
    out_subdir_plots = out_dir / f"violin/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.violinplot(
        data=big_table_subset.query("image_index in @t_range"),
        x="time_hours",
        y="alignment_deg_rel_to_flow",
        ax=ax,
    )
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Alignment (deg)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_alignments_violin.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot alignment vs time as a histogram instead of violinplot
    out_subdir_plots = out_dir / f"alignment_hist/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=big_table_subset,
        x="time_hours",
        y="alignment_deg_rel_to_flow",
        binwidth=(0.5, 1),
        ax=ax,
    )
    ax.set_yticks(range(0, 91, 15))
    ax.set_xlim(0, big_table_subset["time_hours"].max())
    ax.set_ylim(0, 90)
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Alignment (deg)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_alignments_hist.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot the centroid velocities over time as 2D histograms
    out_subdir_plots = out_dir / f"cell_centroid_velocity_orientations/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=big_table_subset,
        x="time_hours",
        y="centroid_velocity_angle_deg",
        binwidth=(0.5, 5),
        ax=ax,
    )
    parallel_angles = [-180, 0, 180]
    perpendicular_angles = [-90, 90]
    for ang in parallel_angles:
        ax.axhline(ang, color="black", linestyle="--", linewidth=1)
    for ang in perpendicular_angles:
        ax.axhline(ang, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(range(-180, 181, 90))
    ax.set_xlim(0, big_table_subset["time_hours"].max())
    ax.set_ylim(-180, 180)
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Centroid Velocity Orientation (deg)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_centroid_velocity_orientations_hist.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot the centroid velocities over time as 2D histograms
    out_subdir_plots = out_dir / f"cell_centroid_velocity_magnitudes/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=big_table_subset,
        x="time_hours",
        y="centroid_velocity_magnitude",
        binwidth=(0.5, None),  # type: ignore[arg-type]
        log_scale=(False, True),
        ax=ax,
    )
    ax.set_xlim(0, big_table_subset["time_hours"].max())
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Centroid Velocity Magnitude (um/minute)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_centroid_velocity_magnitudes_hist.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot alignment vs change in alignment over time
    out_subdir_plots = out_dir / f"alignment_phase/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.scatterplot(
        data=big_table_subset,
        x="alignment_deg_rel_to_flow",
        y="dalignment_dt_deg_rel_to_flow",
        hue="track_id",
        palette="flare",
        alpha=0.5,
        marker=".",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Alignment (deg)")
    ax.set_ylabel("Alignment Change (deg/min)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_alignments_phase.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot alignment vs time with track_id as hue
    out_subdir_plots = out_dir / f"alignments_by_track/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.scatterplot(
        data=big_table_subset,
        x="time_hours",
        y="alignment_deg_rel_to_flow",
        hue="track_id",
        alpha=0.5,
        marker=".",
        lw=0,
        legend=False,
        ax=ax,
    )
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Alignment (deg)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_alignments_by_track.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot distribution orientations of nuclei
    # centroids relative to cell centroids
    out_subdir_plots = out_dir / f"nuc_rel_cell_centroid_angle/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=big_table_subset,
        x="time_hours",
        y="nuc_pos_rel_cell_angle_deg",
        binwidth=(0.5, 5),
        ax=ax,
    )
    parallel_angles = [-180, 0, 180]
    perpendicular_angles = [-90, 90]
    for ang in parallel_angles:
        ax.axhline(ang, color="black", linestyle="--", linewidth=1)
    for ang in perpendicular_angles:
        ax.axhline(ang, color="black", linestyle=":", linewidth=1)
    ax.set_yticks(range(-180, 181, 90))
    ax.set_xlim(0, big_table_subset["time_hours"].max())
    ax.set_ylim(-180, 180)
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Nuclei-Cell Centroid Orientation (deg)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_cell_nuc_orientation.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot distribution magnitudes of nuclei
    # centroids relative to cell centroids
    out_subdir_plots = out_dir / f"nuc_rel_cell_centroid_magnitude/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(AX_WIDTH, AX_HEIGHT))
    sns.histplot(
        data=big_table_subset,
        x="time_hours",
        y="nuc_pos_rel_cell_magnitude",
        binwidth=(0.5, None),  # type: ignore[arg-type]
        log_scale=(False, True),
        ax=ax,
    )
    ax.set_xlim(0, big_table_subset["time_hours"].max())
    ax.set_title(f"{dataset_name} P{position}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Distance Between Nuclei and Cell Centroid (px)")
    plt.tight_layout()
    fig.savefig(
        out_subdir_plots / f"{dataset_name}_P{position}_cell_nuc_dist.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def process_dataset(dataset_name: str, out_dir: Path) -> None:
    # load the segmentation features table
    segprops_manifest = get_live_segmentation_features_manifest([dataset_name])

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
            df_group,
            dataset_name=dataset_nm,
            position=pos,
            out_dir=out_dir,
        )


def main(dataset_name: str | None = None, n_proc: int = 1) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_name)
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
    ipython_cli_flexecute(main)
