from endo_pipeline.settings.workflow_defaults import DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS

TAGS = ["cpu-only"]


def main(
    dataset_collection_name: str = DEFAULT_SEG_FEATURE_WORKFLOW_DATASETS,
    num_plots: int | None = None,
    correlation_var: str = "cos_2_orientation",
    r_max: float = 300,
    bin_width: float = 5,
    wind_threshold: float = 0.35,
    create_movies: bool = False,
):
    """
    Visualize spatial correlation of `correlation_var` over time.
    Default is the orientation of cells relative to flow direction.

    Parameters
    ----------
    dataset_collection_name
        Name of the dataset collection to analyze.
    num_plots
        Number of timepoints to plot for each dataset.
    correlation_var
        Variable to compute spatial correlation for.
    r_max
        Maximum distance to consider for correlation calculation.
    bin_width
        Width of distance bins for correlation calculation.
    wind_threshold
        Threshold for identifying topological defects.
    create_movies
        Whether to create movies from the saved figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    from endo_pipeline.configs.dataset_config_io import get_datasets_in_collection
    from endo_pipeline.io.output import get_output_path
    from endo_pipeline.library.analyze import lib_spatial_correlations as spcorr
    from endo_pipeline.library.process.make_mp4 import make_mp4
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.viz_spatial_correlations import (
        plot_orientational_correlation,
        plot_points_with_angles,
        plot_topological_defects,
    )

    out_path = get_output_path(__file__, correlation_var)

    datasets = get_datasets_in_collection(dataset_collection_name)
    fig_xi, axs_xi = plt.subplots(
        1,
        len(datasets),
        figsize=(len(datasets) * 5, 5),
        dpi=300,
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    for ct, dataset_name in enumerate(datasets):
        dataset_out_path = out_path / dataset_name
        dataset_out_path.mkdir(parents=True, exist_ok=True)

        angle_out_path = dataset_out_path / "points_with_angles"
        angle_out_path.mkdir(parents=True, exist_ok=True)

        corr_out_path = dataset_out_path / "correlation_plots"
        corr_out_path.mkdir(parents=True, exist_ok=True)

        defect_out_path = dataset_out_path / "topological_defects"
        defect_out_path.mkdir(parents=True, exist_ok=True)

        df = spcorr.get_dataframe_for_spatial_correlation_analysis(dataset_name)
        # drop any remaining NaN values
        df = df.dropna(
            subset=[
                "centroid_x_um_on_slide",
                "centroid_y_um_on_slide",
                correlation_var,
                "time_hours",
            ]
        )

        # This is the size of the entire FOV stitched together
        # There is a 10% overlap between positions
        size_x = (
            df["image_size_x"].iloc[0]
            * df["pixel_size_xy_in_um"].iloc[0]
            * df["position"].nunique()
        )
        size_y = df["image_size_y"].iloc[0] * df["pixel_size_xy_in_um"].iloc[0]

        # calculate spatial correlation for all timepoints
        timepoints = np.sort(df["time_hours"].unique())
        if num_plots is None:
            inds_to_plot = np.arange(len(timepoints))
        elif num_plots == 0:
            inds_to_plot = np.array([], dtype=int)
        else:
            inds_to_plot = np.linspace(0, len(timepoints) - 1, num_plots).astype(int)

        correlation_lengthscale_list = []
        for ind, time_hrs in tqdm(
            enumerate(timepoints), total=len(timepoints), desc="Calculating spatial correlation"
        ):
            df_t = df[df["time_hours"] == time_hrs]
            x_t = df_t["centroid_x_um_on_slide"].to_numpy()
            y_t = df_t["centroid_y_um_on_slide"].to_numpy()
            theta_t = df_t["orientation_x"].to_numpy()
            f_t = df_t[correlation_var].to_numpy()

            r_bins = spcorr.make_r_bins(x_t, y_t, r_max=r_max, bin_width=bin_width)
            r, C_r, correlation_lengthscale, _, fitted_c_r = spcorr.pairwise_correlation(
                x_t, y_t, f_t, r_bins
            )

            correlation_lengthscale_list.append(correlation_lengthscale)

            if ind in inds_to_plot:

                # Plot points with angles
                fig_angle, ax_angle = plt.subplots(figsize=(15, 5), dpi=150)
                plot_points_with_angles(
                    x_t, y_t, theta_t, ax=ax_angle, size_x=size_x, size_y=size_y
                )
                # bbox_inches="tight" in `save_plot_to_path` causes
                # issues with unequal figure sizing for stiching into mp4
                fig_angle.savefig(
                    angle_out_path / f"{ind:03d}_{correlation_var}_points_with_angles"
                    f"_time_{time_hrs:0.2f}h_{dataset_name}.png",
                    dpi=150,
                )

                # Plot orientational correlation
                fig_corr, ax_corr = plt.subplots(figsize=(5, 5), dpi=150)
                plot_orientational_correlation(
                    r,
                    C_r,
                    xi_orient=correlation_lengthscale,
                    fitted_c_r=fitted_c_r,
                    ax=ax_corr,
                )
                ax_corr.set_title(
                    f"Dataset: {dataset_name}\n"
                    f"{correlation_var.replace('_', ' ').title()} at {time_hrs:0.2f}h; "
                    f"ξ = {correlation_lengthscale:.2f}"
                )
                fig_corr.savefig(
                    corr_out_path / f"{ind:03d}_{correlation_var}_correlation_time_{time_hrs:0.2f}h"
                    f"_{dataset_name}.png",
                    dpi=150,
                )

                # Plot topological defects
                fig_defects, ax_defects = plt.subplots(figsize=(15, 5), dpi=150)
                defect_positions, defect_numbers, _ = spcorr.calculate_topological_defects(
                    x_t,
                    y_t,
                    theta_t,
                    wind_threshold=wind_threshold,
                )
                plot_topological_defects(
                    x_t,
                    y_t,
                    theta_t,
                    defect_positions,
                    defect_numbers,
                    ax=ax_defects,
                    size_x=size_x,
                    size_y=size_y,
                )
                ax_defects.set_title(
                    f"Topological Defects ({correlation_var.replace('_', ' ').title()})"
                    f" at {time_hrs:0.2f}h"
                )
                fig_defects.savefig(
                    defect_out_path
                    / f"{ind:03d}_{correlation_var}_topological_defects_time_{time_hrs:0.2f}h"
                    f"_{dataset_name}.png",
                    dpi=150,
                )

        # Plot correlation length vs time
        color = get_dataset_color(dataset_name)
        ax_xi = axs_xi[0, ct]
        ax_xi.plot(timepoints, correlation_lengthscale_list, color=color)
        ax_xi.set_xlabel("Time (h)")
        ax_xi.set_ylabel(f"Correlation length ξ ({correlation_var.replace('_', ' ')})")
        ax_xi.set_title(f"Dataset: {dataset_name}")

        # Make movies
        if create_movies:
            for fig_path, label in [
                (angle_out_path, "points_with_angles"),
                (corr_out_path, "correlation_plots"),
                (defect_out_path, "topological_defects"),
            ]:
                make_mp4(
                    str(fig_path),
                    f"{fig_path.parent}/{dataset_name}_{label}.mp4",
                    fps=10,
                )

    fig_xi.savefig(
        out_path / f"{correlation_var}_correlation_length_vs_time.png",
        dpi=300,
    )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
