"""This workflow plots various measures of initial cell density vs shear stress.
Plots are made where data are colored by various different groupings such as
dataset and measures of deviation in orientation.
"""


def main():

    import numpy as np
    from matplotlib import colors
    from matplotlib import pyplot as plt
    from numpy import pi

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.lib_init_density_vs_flow import (
        create_summary_dfs,
        make_summary_plots,
    )
    from endo_pipeline.settings.column_names import ColumnName as Column

    outdir = get_output_path(__file__)

    # get a list of the datasets that will be included in the summary
    datasets = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    # make a list of the columns we need to compute
    dataset_info_cols = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
    ]
    density_cols = [
        Column.SegData.NUM_TRACKS_BEFORE_FILTERING,
        Column.DiffAEData.PC3_FLIPPED,
        Column.SegData.NUM_NUCLEI_IN_CROP,
        Column.SegData.NUM_NUCLEI_AT_TIMEPOINT,
    ]
    filter_cols = [Column.SegDataFilters.IS_INCLUDED]
    feature_cols = [
        Column.SegData.ALIGNMENT,
        Column.SegData.ORIENTATION,
        Column.DiffAEData.POLAR_ANGLE,
    ]
    other_cols = [Column.TRACK_ID, Column.SHEAR_STRESS_REGIME]

    cols_to_compute = dataset_info_cols + density_cols + filter_cols + feature_cols + other_cols

    # create dataframes summarizing some of the columns we are computing
    summary_df_agg, summary_df = create_summary_dfs(datasets, cols_to_compute)

    # define some plotting parameters that will be reused as we plot the data
    # in different ways
    cmap_mag = "inferno"
    hue_norm_mag = colors.Normalize(vmin=0, vmax=1)
    sm_mag = plt.cm.ScalarMappable(cmap=cmap_mag, norm=hue_norm_mag)

    cmap_ang = "hsv"
    hue_norm_angle = colors.Normalize(vmin=0, vmax=pi)
    sm_angle = plt.cm.ScalarMappable(cmap=cmap_ang, norm=hue_norm_angle)

    # these hue groups are the different things that we want to color
    # the datapoints by (along with corresponding parameter values)
    hue_groups_multiposition = [
        (Column.DATASET, "tab20", None, None, True),
        (Column.SHEAR_STRESS_REGIME, "tab10", None, None, True),
        (
            f"{Column.SegData.ORIENTATION}_vec_mean_multipos_magnitude",
            cmap_mag,
            hue_norm_mag,
            sm_mag,
            False,
        ),
        (
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_magnitude",
            cmap_mag,
            hue_norm_mag,
            sm_mag,
            False,
        ),
        (
            f"{Column.SegData.ORIENTATION}_vec_mean_multipos_angle",
            cmap_ang,
            hue_norm_angle,
            sm_angle,
            False,
        ),
        (
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_multipos_angle",
            cmap_ang,
            hue_norm_angle,
            sm_angle,
            False,
        ),
    ]

    hue_groups_single_position = [
        (f"{Column.SegData.ORIENTATION}_vec_mean_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
        (
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_magnitude",
            cmap_mag,
            hue_norm_mag,
            sm_mag,
            False,
        ),
        (f"{Column.SegData.ORIENTATION}_vec_mean_angle", cmap_ang, hue_norm_angle, sm_angle, False),
        (
            f"{Column.DiffAEData.POLAR_ANGLE}_vec_mean_angle",
            cmap_ang,
            hue_norm_angle,
            sm_angle,
            False,
        ),
    ]

    # make a bunch of plots for each of our different density metrics to
    for dens_col in density_cols:

        # make a summary plot for each of the things we are coloring datapoints
        # by where each position in a dataset is a data point
        for hue_col, cmap, norm, cbar, legend in hue_groups_single_position:
            out_subdir = outdir / "single_position"
            out_subdir.mkdir(parents=True, exist_ok=True)

            make_summary_plots(
                out_dir=out_subdir,
                filename=f"{hue_col}_vs_{dens_col}_vs_flow",
                df=summary_df.dropna(subset=hue_col),
                x="shear_stress",
                x_label="Shear Stress (dyn/cm²)",
                y=dens_col,
                hue=hue_col,
                hue_norm=norm,
                cmap=cmap,
                cbar_scalarmap=cbar,
                legend=legend,
            )

        # make a summary plot for each of the things we are coloring datapoints
        # by where positions are aggregated and each data point is a dataset
        for hue_col, cmap, norm, cbar, legend in hue_groups_multiposition:
            out_subdir = outdir / "multiposition"
            out_subdir.mkdir(parents=True, exist_ok=True)

            make_summary_plots(
                out_dir=out_subdir,
                filename=f"{hue_col}_vs_{dens_col}_vs_flow",
                df=summary_df_agg,
                x="shear_stress",
                x_label="Shear Stress (dyn/cm²)",
                y=dens_col,
                hue=hue_col,
                hue_norm=norm,
                cmap=cmap,
                cbar_scalarmap=cbar,
                legend=legend,
            )

    # plot hue vectors as quiver plots
    for dens_col in density_cols:

        out_subdir = outdir / "quiver_plots"
        out_subdir.mkdir(parents=True, exist_ok=True)

        for col_prefix in [Column.SegData.ORIENTATION, Column.DiffAEData.POLAR_ANGLE]:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

            hue_mag_col = f"{col_prefix}_vec_mean_multipos_magnitude"
            hue_angle_col = f"{col_prefix}_vec_mean_multipos_angle"
            x_values = summary_df_agg["shear_stress"].to_numpy()
            y_values = summary_df_agg[dens_col].to_numpy()
            magnitudes = summary_df_agg[hue_mag_col].to_numpy()
            angles = summary_df_agg[hue_angle_col].to_numpy()

            u_values = magnitudes * np.cos(angles)
            v_values = magnitudes * np.sin(angles)

            ax.quiver(
                x_values,
                y_values,
                u_values,
                v_values,
                magnitudes,
                angles="uv",
                scale=10,
                headwidth=3,
                headlength=3,
                headaxislength=3,
                minlength=0.1,
            )
            ax.set_xlabel("Shear Stress (dyn/cm²)")
            ax.set_ylabel(dens_col)
            ax.set_title(f"Quiver plot of {col_prefix} vectors")
            fig.savefig(
                out_subdir / f"quiver_{col_prefix}_{dens_col}_vs_flow.png", bbox_inches="tight"
            )
            plt.show()
            plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
