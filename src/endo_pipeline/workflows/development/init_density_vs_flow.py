"""This workflow plots various measures of initial cell density vs shear stress.
Plots are made where data are colored by various different groupings such as
dataset and measures of deviation in orientation.
"""


def main():

    from matplotlib import colors
    from matplotlib import pyplot as plt
    from numpy import pi

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.lib_init_density_vs_flow import (
        create_summary_dfs,
        make_summary_plots,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES, ColumnName

    outdir = get_output_path(__file__)

    # get a list of the datasets that will be included in the summary
    datasets = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")

    # make a list of the columns we need to compute
    dataset_info_cols = [
        ColumnName.DATASET.value,
        ColumnName.POSITION.value,
        ColumnName.TIMEPOINT.value,
    ]
    density_cols = [
        "num_unique_tracks_before_filtering_at_T",
        DIFFAE_PC_COLUMN_NAMES[2],
        "num_nuclei_in_crop",
        "total_nuclei_count_at_T",
    ]
    filter_cols = ["is_included"]
    feature_cols = ["alignment_rel_to_flow", "orientation", "polar_theta"]
    other_cols = ["track_id", "shear_stress_regime"]

    cols_to_compute = dataset_info_cols + density_cols + filter_cols + feature_cols + other_cols

    # create dataframes summarizing some of the columns we are computing
    summary_df_agg, summary_df = create_summary_dfs(datasets, cols_to_compute)[:2]

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
        (ColumnName.DATASET.value, "tab20", None, None, True),
        ("shear_stress_regime", "tab10", None, None, True),
        ("orientation_vec_mean_multipos_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
        ("polar_theta_vec_mean_multipos_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
        ("orientation_vec_mean_multipos_angle", cmap_ang, hue_norm_angle, sm_angle, False),
        ("polar_theta_vec_mean_multipos_angle", cmap_ang, hue_norm_angle, sm_angle, False),
    ]

    hue_groups_single_position = [
        ("orientation_vec_mean_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
        ("polar_theta_vec_mean_magnitude", cmap_mag, hue_norm_mag, sm_mag, False),
        ("orientation_vec_mean_angle", cmap_ang, hue_norm_angle, sm_angle, False),
        ("polar_theta_vec_mean_angle", cmap_ang, hue_norm_angle, sm_angle, False),
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

    print("Done.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
