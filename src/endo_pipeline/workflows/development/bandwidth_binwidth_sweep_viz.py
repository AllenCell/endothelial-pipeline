"""
Visualization script for bandwidth and binwidth sweep results.
"""


def main():

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.bandwidth_binwidth_sweep import (
        get_param_sweep_run_name,
        get_parameter_space,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings import plot_defaults
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName, ColumnNameTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        POLAR_ANGLE_PERIOD,
    )
    from endo_pipeline.settings.figures import FONTSIZE_MEDIUM, FONTSIZE_SMALL, FONTSIZE_XSMALL
    from endo_pipeline.settings.manifest_names import GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME

    out_dir = get_output_path(__file__)

    # start a single big dataframe to hold the data for multiple datasets
    big_df = pd.DataFrame()

    parameter_space = get_parameter_space()
    for bw, kb in parameter_space:
        sweep_run_name = get_param_sweep_run_name(bw, kb)
        manifest = load_dataframe_manifest(
            f"{GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME}_{sweep_run_name}"
        )
        # skip the manifest if it has no locations (i.e. no bootstrapped fixed points were found)
        if not manifest.locations:
            continue

        for dataset in manifest.locations:
            df = load_dataframe(manifest.locations[dataset])
            # add the parameter values to the dataframe
            df["bw"] = bw
            df["kb"] = kb

            # add the dataframe for this dataset to the single big dataframe if it is not empty, otherwise
            # initialize it with the current dataframe
            if big_df.empty:
                big_df = df
            else:
                big_df = pd.concat([big_df, df], ignore_index=True)

    column_names = big_df.columns
    unwrap_angle = True
    if ColumnNameTemplate.BASELINE_FIXED_POINT % ColumnName.DiffAEData.POLAR_ANGLE in column_names:
        if unwrap_angle:
            # unwrap baseline, bootstrapped cluster mean angle, and CI bounds
            for template in [
                ColumnNameTemplate.BASELINE_FIXED_POINT,
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN,
                ColumnNameTemplate.BOOTSTRAP_CI_LOWER,
                ColumnNameTemplate.BOOTSTRAP_CI_UPPER,
            ]:
                angle_column = template % ColumnName.DiffAEData.POLAR_ANGLE
                big_df[angle_column] = big_df[angle_column].apply(
                    lambda angle: (angle - POLAR_ANGLE_PERIOD if angle > (5 * np.pi / 6) else angle)
                )

    for dataset, df in big_df.groupby(ColumnName.DATASET):

        df_filtered = df.loc[
            df[ColumnName.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
        ].copy()

        fixed_point_stabilities: list[str] = (
            df_filtered[ColumnName.FIXED_POINT_STABILITY].unique().tolist()
        )

        # create plots for the stable and unstable fixed points
        for stability in fixed_point_stabilities:
            # count the number of fixed points for this stability that passed the bootstrap threshold
            df_agg = (
                df_filtered.assign(
                    _matches_stability=df_filtered[ColumnName.FIXED_POINT_STABILITY].eq(stability)
                )
                .groupby(["kb", "bw"], observed=False)["_matches_stability"]
                .sum()
                .reset_index(name="num_fixedpoints")
            )

            # plot the heatmap
            fig, ax = plt.subplots(figsize=(2.2, 2))
            sns.heatmap(
                df_agg.pivot(index="bw", columns="kb", values="num_fixedpoints"),
                cmap="magma",
                linewidths=0.5,
                annot=True,
                annot_kws={"fontsize": FONTSIZE_XSMALL},
                ax=ax,
                cbar_kws={"label": f"Number of {stability.capitalize()}\nFixed Points"},
                vmin=0,
            )

            # adjust the ticklabel fontsizes
            ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_SMALL)

            # adjust the colorbar to drop the tickmarks at the boundaries between colors
            cbar = ax.collections[0].colorbar
            if cbar is not None:
                cbar.ax.minorticks_off()
                cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)
                cbar.set_label(cbar.ax.get_ylabel(), fontsize=FONTSIZE_SMALL)
                cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

            # the smallest binwidth is defaulting to the top of the heatmap for some reason, so invert
            # the y-axis so that the smallest binwidth appears at the bottom
            ax.invert_yaxis()

            # give the heatmap a title and clean labels
            ax.set_title(str(dataset), fontsize=FONTSIZE_MEDIUM)
            ax.set_xlabel("Kernel Bandwidth", fontsize=FONTSIZE_SMALL)
            ax.set_ylabel("Binwidth", fontsize=FONTSIZE_SMALL)

            # save the heatmap to the results folder
            save_plot_to_path(
                figure=fig,
                output_path=out_dir,
                figure_name=f"{dataset}_{stability}_heatmap.png",
            )

            # clean up the dataframe for summary plots:
            # create a new column with the parameter combination as a label
            df_filtered["param_combo"] = df_filtered.apply(
                lambda row: f"(kb={row['kb']}, bw={row['bw']})", axis=1
            )

            # rename columns to use the pretty column names
            columns_for_summary_plots = {
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.POLAR_ANGLE: ColumnName.DiffAEData.POLAR_ANGLE,
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.POLAR_RADIUS: ColumnName.DiffAEData.POLAR_RADIUS,
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.PC3_FLIPPED: ColumnName.DiffAEData.PC3_FLIPPED,
            }
            df_for_summary_plots = df_filtered.rename(
                columns=columns_for_summary_plots, inplace=False
            )
            df_for_summary_plots = df_for_summary_plots[
                df_for_summary_plots[ColumnName.FIXED_POINT_STABILITY] == stability
            ]
            df_for_summary_plots.sort_values(by=["kb", "bw"], inplace=True)
            # do a scatterplot of the fixed points at each ML-based feature for each dataset
            fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(6.3, 4.0), sharex=True)
            for col, ax in zip(DYNAMICS_COLUMN_NAMES, axes, strict=True):
                # err_low_col = ColumnNameTemplate.BOOTSTRAP_CI_LOWER % col
                # err_high_col = ColumnNameTemplate.BOOTSTRAP_CI_UPPER % col
                # y_errors = (
                #     df_for_summary_plots[col] - df_for_summary_plots[err_low_col],
                #     df_for_summary_plots[err_high_col] - df_for_summary_plots[col],
                # )
                ax.scatter(
                    x=df_for_summary_plots["param_combo"],
                    y=df_for_summary_plots[col],
                    alpha=0.5,
                    c="black",
                    marker=".",
                    ls="",
                )
                # ax.errorbar(
                #     x=df_for_summary_plots["param_combo"],
                #     y=df_for_summary_plots[col],
                #     yerr=y_errors,
                #     alpha=0.5,
                #     c="black",
                #     marker=".",
                #     ls="",
                # )
                column_metadata = COLUMN_METADATA[col]
                ax.set_ylabel(f"{column_metadata.label_with_unit}$^*$", fontsize=FONTSIZE_SMALL)
                if col == ColumnName.DiffAEData.POLAR_ANGLE:
                    ax.set_ylim(plot_defaults.SUMMARY_PLOT_THETA_RANGE)
                else:
                    ax.set_ylim(column_metadata.min, column_metadata.max)
                if column_metadata.ticks is not None:
                    ax.set_yticks(column_metadata.ticks)
                if column_metadata.tick_labels is not None:
                    ax.set_yticklabels(column_metadata.tick_labels, fontsize=FONTSIZE_SMALL)

                # lastly, draw a line where the fixed point parameters used in the original analysis are
                bw_original = np.unique(list(BIN_WIDTHS_DYNAMICS.values())).item()
                kb_original = np.unique(list(KERNEL_BANDWIDTHS_DYNAMICS.values())).item()
                ax.axvline(
                    x=f"(kb={kb_original}, bw={bw_original})",
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )

            axes[-1].set_xticklabels(
                axes[-1].get_xticklabels(), rotation=45, ha="right", fontsize=FONTSIZE_SMALL
            )
            axes[-1].set_xlabel("Parameter Combination", fontsize=FONTSIZE_SMALL)
            fig.suptitle(
                f"Fixed Point Positions for {dataset} ({stability})", fontsize=FONTSIZE_MEDIUM
            )

            # save the scatterplot to the results folder
            save_plot_to_path(
                figure=fig,
                output_path=out_dir,
                figure_name=f"{dataset}_{stability}_fixed_point_positions.png",
            )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
