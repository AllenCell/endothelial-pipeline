from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from cellsmap.util.dataset_io import (
    fire_parse_generate_dataset_name_list,
    get_segmentation_features_manifest,
    ipython_cli_flexecute,
    save_git_versioning_info,
)
from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.workflows.make_seg_feats_manifest import (
    calculate_derived_data_dynamics_dependent,
)


def plot_per_position(
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
    num_positions = df_group["position"].nunique()
    assert (
        len(df_group["dataset_name"].unique()) == 1
    ), f'Only a single dataset allowed in df_group, datasets found: {df_group["dataset_name"].unique()}'
    dataset_name = df_group["dataset_name"].unique()[0]
    assert (
        len(df_group["position"].unique()) == 1
    ), f'Only a single position allowed in df_group, position found: {df_group["position"].unique()}'
    position = df_group["position"].unique()[0]

    ax_height = 6
    ax_width = 6 * (1 + 5 ** (1 / 2)) / 2

    fig, ax = plt.subplots(nrows=num_positions, figsize=(ax_width, ax_height * num_positions))
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


def plot_tracking_data(
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
            (None, None),
            f"{dataset_name}_P{position}_nematic_order.png",
        ),
        (
            "time_hours",
            "aspect_ratio",
            "Time (hours)",
            "Aspect Ratio",
            (None, None),
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
            "centroid_velocity_angle_deg_rel_to_flow",
            "Time (hours)",
            "Centroid Velocity Alignment (deg)",
            (0, 90),
            f"{dataset_name}_P{position}_centroid_velocity_angles.png",
        ),
        (
            "time_hours",
            "centroid_velocity_magnitude",
            "Time (hours)",
            "Centroid Velocity Magnitude (px/frame)",
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
        plot_per_position(
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

    # plot alignment vs change in alignment over time
    out_subdir_plots = out_dir / f"alignment_phase/{dataset_name}"
    out_subdir_plots.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
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


def process_dataset(dataset_name: str, out_dir: Path) -> None:
    # load the segmentation features table
    segprops_manifest = get_segmentation_features_manifest([dataset_name])

    # apply the data filter
    segprops_manifest = segprops_manifest[~segprops_manifest["filter_global"]]

    # make basic plots for each dataset
    for (dataset_nm, pos), df_group in tqdm(
        segprops_manifest.groupby(["dataset_name", "position"]),
        total=len(segprops_manifest.groupby(["dataset_name", "position"])),
        desc=f"Plotting features: {dataset_name}",  # type: ignore
        unit="position",
    ):
        # calculate the dynamics-dependent features
        df_group = calculate_derived_data_dynamics_dependent(df_group)

        # make some plots
        plot_tracking_data(
            df_group,
            dataset_name=dataset_nm,
            position=pos,
            out_dir=out_dir,
        )


def main(dataset_names: str | None = None) -> None:

    dataset_name_list = fire_parse_generate_dataset_name_list(dataset_names)

    out_dir = Path(get_output_path(Path(__file__).stem, verbose=True))
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in dataset_name_list:
        # process dataset below will both load and plot the data
        process_dataset(dataset, out_dir)

    # save git versioning info
    save_git_versioning_info(out_dir, Path(__file__).stem, verbose=False)


if __name__ == "__main__":
    ipython_cli_flexecute(main)
