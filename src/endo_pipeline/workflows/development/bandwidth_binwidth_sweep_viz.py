"""
Visualization script for bandwidth and binwidth sweep results.
"""


def main():
    """Plot the kernel bandwidth and binwidth sweep results."""
    from itertools import product

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.bandwidth_binwidth_sweep import (
        get_param_sweep_run_name,
        get_parameter_space,
    )
    from endo_pipeline.manifests import load_dataframe_manifest
    from endo_pipeline.settings import plot_defaults
    from endo_pipeline.settings.bandwidth_binwidth_sweep import (
        SWEEP_PARAM_EXAMPLE_BINWIDTHS,
        SWEEP_PARAM_EXAMPLE_KERNEL_BANDWIDTHS,
    )
    from endo_pipeline.settings.bootstrap_fixed_points import BOOTSTRAP_THRESHOLD
    from endo_pipeline.settings.column_metadata import COLUMN_METADATA
    from endo_pipeline.settings.column_names import ColumnName, ColumnNameTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_WIDTHS_DYNAMICS,
        DYNAMICS_COLUMN_NAMES,
        KERNEL_BANDWIDTHS_DYNAMICS,
        POLAR_ANGLE_PERIOD,
    )
    from endo_pipeline.settings.examples import SUPP_FIG_PARAM_SWEEP
    from endo_pipeline.settings.figures import (
        FONTSIZE_MEDIUM,
        FONTSIZE_SMALL,
        FONTSIZE_XSMALL,
        MAX_FIGURE_WIDTH,
    )
    from endo_pipeline.settings.manifest_names import GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME

    out_dir = get_output_path(__file__)

    file_format = ".png"

    # Get the parameters that surround the one used in the main analysis (including the main
    # analysis parameters)
    local_parameter_space = list(
        product(SWEEP_PARAM_EXAMPLE_BINWIDTHS, SWEEP_PARAM_EXAMPLE_KERNEL_BANDWIDTHS)
    )

    # start a single big dataframe to hold the data for multiple datasets
    big_df = pd.DataFrame()

    parameter_space = get_parameter_space()
    # the full grid of all binwidths and kernel bandwidths used anywhere in the sweep, used to lay out
    # the heatmaps so that every combination gets a square
    sweep_binwidths_all = sorted({bw for bw, _ in parameter_space})
    sweep_kernel_bandwidths_all = sorted({kb for _, kb in parameter_space})
    for bw, kb in parameter_space:
        sweep_run_name = get_param_sweep_run_name(bw, kb)
        manifest = load_dataframe_manifest(
            f"{GRID_BASED_BOOTSTRAPPING_MANIFEST_NAME}_{sweep_run_name}"
        )
        # skip the manifest if it has no locations (i.e. no bootstrapped fixed points were found)
        # and record the unsuccessful parameter combination, they will be added as nan values later
        if not manifest.locations:
            print(f"Skipping parameters bw={bw}, kb={kb} due to empty manifest.")
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

    # Filter to only include datasets specified in SUPP_FIG_PARAM_SWEEP
    df_filtered = big_df.loc[big_df[ColumnName.DATASET].isin(SUPP_FIG_PARAM_SWEEP)].copy()

    # shared horizontal layout for the heatmap and fixed point location figures (all in inches) so
    # that their dataset columns line up when the separately saved figures are stacked: the four
    # columns of axes span `panel_block_width`, the colorbar of the heatmap figure is added in extra
    # space to the right of that block. The block width is chosen so that the heatmap figure
    # including its colorbar stays within MAX_FIGURE_WIDTH
    panel_block_width = 5.6
    left_margin = 0.55
    right_margin = 0.05
    wspace = 0.1
    cbar_pad = 0.08
    cbar_width = 0.08
    cbar_label_space = 0.5

    heatmap_fig_height = 1.9
    heatmap_bottom_margin = 0.65
    heatmap_top_margin = 0.25
    heatmap_fig_width = panel_block_width + cbar_pad + cbar_width + cbar_label_space
    assert heatmap_fig_width <= MAX_FIGURE_WIDTH

    fp_loc_fig_height = 3.5
    fp_loc_bottom_margin = 1.2
    fp_loc_top_margin = 0.25

    for stability, df_stability in df_filtered.groupby(ColumnName.FIXED_POINT_STABILITY):

        # the stability column holds string labels, narrow the type with this assertion for mypy
        assert isinstance(stability, str)

        # skip this stability if no parameter combination has a detection rate below the threshold
        if not (df_stability[ColumnName.FIXED_POINT_DETECTION_RATE] < BOOTSTRAP_THRESHOLD).any():
            continue

        # create plots for each stability in fixed_point_stabilities; the four heatmap axes are laid
        # out with the shared horizontal layout and a colorbar axis is added in the extra space to
        # the right
        fig_heatmaps, heatmap_axes = plt.subplots(
            figsize=(heatmap_fig_width, heatmap_fig_height), ncols=4, sharey=True
        )
        heatmap_bottom = heatmap_bottom_margin / heatmap_fig_height
        heatmap_top = 1 - heatmap_top_margin / heatmap_fig_height
        fig_heatmaps.subplots_adjust(
            left=left_margin / heatmap_fig_width,
            right=(panel_block_width - right_margin) / heatmap_fig_width,
            bottom=heatmap_bottom,
            top=heatmap_top,
            wspace=wspace,
        )
        cbar_ax = fig_heatmaps.add_axes(
            (
                (panel_block_width + cbar_pad) / heatmap_fig_width,
                heatmap_bottom,
                cbar_width / heatmap_fig_width,
                heatmap_top - heatmap_bottom,
            )
        )

        # use a common color range across all datasets so that the shared colorbar applies to every
        # heatmap
        max_num_fixedpoints = (
            df_stability.loc[
                df_stability[ColumnName.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD
            ]
            .groupby([ColumnName.DATASET, "kb", "bw"], observed=True)
            .size()
            .max()
        )
        vmax = max(int(max_num_fixedpoints) if pd.notna(max_num_fixedpoints) else 0, 1)

        for dataset, df in df_stability.groupby(ColumnName.DATASET):
            # plot the heatmaps in the same order as defined in SUPP_FIG_PARAM_SWEEP
            # (low, high, intermediate bistable, intermediate monostable)
            axis_index = SUPP_FIG_PARAM_SWEEP.index(str(dataset))
            ax = heatmap_axes[axis_index]

            # count the number of fixed points for this stability that passed the bootstrap threshold
            df_agg = (
                df.loc[df[ColumnName.FIXED_POINT_DETECTION_RATE] >= BOOTSTRAP_THRESHOLD]
                .groupby(["kb", "bw"], observed=False)
                .size()
                .reset_index(name="num_fixedpoints")
            )

            # remove the axes instance at this axis_index if df_agg is empty
            if df_agg.empty:
                fig_heatmaps.delaxes(ax)
                continue

            # lay the counts out on the full grid of all sweep parameters so that every parameter
            # combination appears in the heatmap
            heatmap_data = df_agg.pivot(index="bw", columns="kb", values="num_fixedpoints").reindex(
                index=sweep_binwidths_all,
                columns=sweep_kernel_bandwidths_all,
            )

            # parameter combinations that produced results for this dataset but had no fixed points of
            # this stability pass the threshold are counted as 0; combinations with no results at all
            # are left as NaN so that they are drawn in grey (NaN cells are left unpainted by seaborn,
            # letting the axes background show through)
            successful_combos = big_df.loc[
                big_df[ColumnName.DATASET] == dataset, ["bw", "kb"]
            ].drop_duplicates()
            for bw, kb in successful_combos.itertuples(index=False):
                if bw in heatmap_data.index and kb in heatmap_data.columns:
                    if pd.isna(heatmap_data.at[bw, kb]):
                        heatmap_data.at[bw, kb] = 0

            # plot a heatmap
            ax.set_facecolor("lightgrey")
            sns.heatmap(
                heatmap_data,
                cmap="magma",
                linewidths=0.5,
                annot=True,
                annot_kws={"fontsize": FONTSIZE_XSMALL},
                ax=ax,
                vmin=0,
                vmax=vmax,
                cbar=False,
                # force a tick and label at every parameter value rather than letting seaborn thin
                # them out to avoid overlap
                xticklabels=[f"{kb:g}" for kb in heatmap_data.columns],
                yticklabels=[f"{bw:g}" for bw in heatmap_data.index],
            )

            # adjust the ticklabel fontsizes and rotate the x ticklabels so they don't overlap
            ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_XSMALL)
            ax.tick_params(axis="x", rotation=90)
            ax.tick_params(axis="y", rotation=0)

            # give the plot clean labels
            ax.set_title(str(dataset), fontsize=FONTSIZE_MEDIUM)
            ax.set_xlabel("Kernel Bandwidth", fontsize=FONTSIZE_SMALL)
            if axis_index == 0:
                ax.set_ylabel("Binwidth", fontsize=FONTSIZE_SMALL)
            else:
                ax.set(ylabel=None)

        # draw a single colorbar shared by all heatmaps into the last axis, dropping the tickmarks
        # at the boundaries between colors
        cbar = fig_heatmaps.colorbar(
            ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap="magma"), cax=cbar_ax
        )
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(labelsize=FONTSIZE_XSMALL)
        cbar.set_label(f"Number of {stability.capitalize()}\nFixed Points", fontsize=FONTSIZE_SMALL)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

        # save the plot to the results folder; the layout is set explicitly above, so skip tight
        # layout / bbox cropping that would shift the axes
        save_plot_to_path(
            figure=fig_heatmaps,
            output_path=out_dir,
            figure_name=f"{stability}_heatmap",
            file_format=file_format,
            pad_inches=0,
            tight_layout=False,
        )

        # share the y-axis within each row (one feature per row) and the x-axis within each column,
        # and use the shared horizontal layout so the columns line up with the heatmap figure
        fig_fixed_point_location, axes_fixed_point_location = plt.subplots(
            figsize=(panel_block_width, fp_loc_fig_height),
            ncols=4,
            nrows=3,
            sharex=True,
            sharey="row",
        )
        fig_fixed_point_location.subplots_adjust(
            left=left_margin / panel_block_width,
            right=(panel_block_width - right_margin) / panel_block_width,
            bottom=fp_loc_bottom_margin / fp_loc_fig_height,
            top=1 - fp_loc_top_margin / fp_loc_fig_height,
            wspace=wspace,
        )

        for dataset, df in df_stability.groupby(ColumnName.DATASET):
            axis_index = SUPP_FIG_PARAM_SWEEP.index(str(dataset))
            fp_loc_axes = axes_fixed_point_location[:, axis_index]

            # label each column of axes with the dataset name over the top row only
            fp_loc_axes[0].set_title(str(dataset), fontsize=FONTSIZE_MEDIUM)

            # clean up the dataframe for summary plots:
            # subset the dataframe to only include the parameter combinations in the local parameter space
            df = df[df.apply(lambda row: (row["bw"], row["kb"]) in local_parameter_space, axis=1)]
            # create a new column with the parameter combination as a label
            df["param_combo"] = df.apply(lambda row: f"({row['kb']}, {row['bw']})", axis=1)

            # rename columns to use the pretty column names
            columns_for_summary_plots = {
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.POLAR_ANGLE: ColumnName.DiffAEData.POLAR_ANGLE,
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.POLAR_RADIUS: ColumnName.DiffAEData.POLAR_RADIUS,
                ColumnNameTemplate.BOOTSTRAP_CLUSTER_MEAN
                % ColumnName.DiffAEData.PC3_FLIPPED: ColumnName.DiffAEData.PC3_FLIPPED,
            }
            df_for_summary_plots = df.rename(columns=columns_for_summary_plots, inplace=False)
            df_for_summary_plots = df_for_summary_plots[
                df_for_summary_plots[ColumnName.FIXED_POINT_STABILITY] == stability
            ]
            df_for_summary_plots.sort_values(by=["kb", "bw"], inplace=True)

            # do a scatterplot of the fixed points at each ML-based feature for each dataset
            for col, ax in zip(DYNAMICS_COLUMN_NAMES, fp_loc_axes, strict=True):
                ax.scatter(
                    x=df_for_summary_plots["param_combo"],
                    y=df_for_summary_plots[col],
                    alpha=0.5,
                    c="black",
                    marker=".",
                    ls="",
                )
                column_metadata = COLUMN_METADATA[col]
                if col == ColumnName.DiffAEData.POLAR_ANGLE:
                    ax.set_ylim(plot_defaults.SUMMARY_PLOT_THETA_RANGE)
                else:
                    ax.set_ylim(column_metadata.min, column_metadata.max)
                if column_metadata.ticks is not None:
                    ax.set_yticks(column_metadata.ticks)
                if column_metadata.tick_labels is not None:
                    ax.set_yticklabels(column_metadata.tick_labels)

                # only show the y-axis label and ticklabels on the first column
                if axis_index == 0:
                    ax.set_ylabel(f"{column_metadata.label_with_unit}$^*$", fontsize=FONTSIZE_SMALL)
                    ax.tick_params(axis="y", labelsize=FONTSIZE_XSMALL)
                else:
                    ax.set(ylabel=None)
                    ax.tick_params(axis="y", labelleft=False)

                # lastly, draw a line where the fixed point parameters used in the original analysis are
                bw_original = np.unique(list(BIN_WIDTHS_DYNAMICS.values())).item()
                kb_original = np.unique(list(KERNEL_BANDWIDTHS_DYNAMICS.values())).item()
                ax.axvline(
                    x=f"({kb_original}, {bw_original})",
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    zorder=0,
                )

            fp_loc_axes[-1].set_xticklabels(
                fp_loc_axes[-1].get_xticklabels(), rotation=90, fontsize=FONTSIZE_XSMALL
            )
            fp_loc_axes[-1].set_xlabel("Parameters (kb, bw)", fontsize=FONTSIZE_SMALL)

        # save the scatterplot to the results folder; the layout is set explicitly above, so skip
        # tight layout that would shift the axes
        save_plot_to_path(
            figure=fig_fixed_point_location,
            output_path=out_dir,
            figure_name=f"{stability}_fixed_point_positions",
            file_format=file_format,
            pad_inches=0,
            tight_layout=False,
        )


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
