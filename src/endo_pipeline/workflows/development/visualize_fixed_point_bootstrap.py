from endo_pipeline.cli import CropPattern, Datasets


def main(
    crop_pattern: CropPattern = "grid",
    datasets: Datasets | None = None,
    bootstrap_threshold: float = 0.5,
) -> None:
    """
    Visualize bootstrap-validated fixed points from `bootstrap-fixed-points`.

    #dynamical-systems #diffae-feature-analysis #bootstrap

    **Overview**

    This workflow loads the bootstrap confidence interval (CI) dataframes
    produced by `generate-3d-flow-field-bootstrap`, filters to fixed points
    whose *bootstrap detection rate* meets or exceeds `bootstrap_threshold`, and
    plots their locations with per-coordinate confidence interval error bars.

    **Filtering**

    Only fixed points with `bootstrap_detection_rate >= bootstrap_threshold` are
    retained. The `bootstrap_detection_rate` is the fraction of bootstrap
    iterations in which a fixed point was detected within the specified
    `bootstrap_match_radius` of a baseline fixed point.

    **Visualizations**

    This workflow produces a two-panel plot for each dataset, showing the
    high-confidence fixed points in two 2D projections:

    - Top panel: PC1 (polar angle) vs PC2 (polar radius)
    - Bottom panel: PC1 (polar angle) vs PC3 (PC3-flipped)

    Fixed points are drawn as scatter markers coloured by stability
    classification (stable = blue circle, saddle = grey triangle, unstable = red
    square, indeterminate = yellow plus); markers are placed at the bootstrap
    cluster mean coordinate (mean of all matched bootstrap fixed point
    coordinates) and error bars show the per-coordinate bootstrap CIs at the
    percentiles used during the bootstrap run (`FP_CI_LOWER_PERCENTILE` and
    `FP_CI_UPPER_PERCENTILE`).

    If fewer than two bootstrap hits were obtained for a fixed point the CI is
    `nan` and no error bar is drawn. If high-confidence fixed points are found
    for two or more datasets a combined comparison figure coloured by dataset is
    also saved.

    Parameters
    ----------
    crop_pattern
        The crop pattern to load bootstrap fixed point dataframes for.
    datasets
        Optional list of specific datasets to visualize.
    bootstrap_threshold
        Minimum bootstrap detection rate for a fixed point to be included in the plots.

    """
    import logging

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_by_shear_stress
    from endo_pipeline.library.visualize.diffae_features.feature_viz import (
        get_dataset_color,
        get_label_for_column,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        BIN_LIMITS_DYNAMICS,
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
    )
    from endo_pipeline.settings.flow_field_3d import FIGSIZE_2D_FLOW_FIELD, NROWS_2D_FLOW_FIELD
    from endo_pipeline.settings.flow_field_dataframes import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        StabilityLabel,
    )
    from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE, StabilityLegendHandle
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode

    logger = logging.getLogger(__name__)

    column_names: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES)

    bootstrap_fp_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[crop_pattern]

    # Flexible DEMO_MODE loading pattern: try without demo suffix first so this
    # workflow can visualise a full production run even when DEMO_MODE is set.
    # Fall back to the demo-suffixed manifest only in DEMO_MODE.
    try:
        bootstrap_fp_manifest = load_dataframe_manifest(bootstrap_fp_manifest_name)
    except FileNotFoundError:
        if DEMO_MODE:
            fallback_name = f"{BOOTSTRAPPING_MANIFEST_NAMES[crop_pattern]}_demo"
            logger.warning(
                "Bootstrap fixed point manifest [ %s ] not found; trying [ %s ].",
                bootstrap_fp_manifest_name,
                fallback_name,
            )
            bootstrap_fp_manifest = load_dataframe_manifest(fallback_name)
        else:
            raise

    n_bootstrap = bootstrap_fp_manifest.parameters.get("num_bootstrap_iterations")
    if n_bootstrap is None:
        logger.warning(
            "Number of bootstrap samples not found in manifest parameters; "
            "bootstrap detection rates will be included in the plots but "
            "not the number of bootstrap samples."
        )

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)
    if DEMO_MODE:
        logger.warning("DEMO MODE: Processing no more than two datasets for quick visualization.")
        dataset_names = dataset_names[: min(len(dataset_names), 2)]

    fig_savedir = get_output_path(__file__, crop_pattern)

    # Axis bounds from global bin limits, one tuple (min, max) per column
    bounds_for_plots = BIN_LIMITS_DYNAMICS.copy()

    all_high_confidence_dfs: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Per-dataset loop
    # ------------------------------------------------------------------
    for dataset_name in dataset_names:
        if dataset_name not in bootstrap_fp_manifest.locations:
            logger.warning(
                "No bootstrap fixed point dataframe found in manifest [ %s ] for dataset "
                "[ %s ]. Skipping.",
                bootstrap_fp_manifest.name,
                dataset_name,
            )
            continue

        dataset_config = load_dataset_config(dataset_name)
        logger.info("Loading bootstrap fixed point dataframe for dataset [ %s ].", dataset_name)
        location = get_dataframe_location_for_dataset(bootstrap_fp_manifest, dataset_name)
        df = load_dataframe(location, delay=False)

        n_total = len(df)
        high_confidence_df = df[
            df[Column.BootstrapAnalysis.DETECTION_RATE] >= bootstrap_threshold
        ].copy()

        if high_confidence_df.empty:
            logger.warning(
                "No fixed points with bootstrap_detection_rate >= %.2f for dataset "
                "[ %s ] (%d fixed points in total). Skipping plot for this dataset.",
                bootstrap_threshold,
                dataset_name,
                n_total,
            )
            continue

        logger.info(
            "Dataset [ %s ]: %d / %d fixed point(s) pass bootstrap threshold (>= %.2f).",
            dataset_name,
            len(high_confidence_df),
            n_total,
            bootstrap_threshold,
        )

        all_high_confidence_dfs.append(high_confidence_df)

        # Per-dataset, per flow condition figures
        for flow_condition in dataset_config.flow_conditions:
            shear_stress = flow_condition.shear_stress

            high_confidence_df_flow = filter_dataframe_by_shear_stress(
                high_confidence_df, shear_stress
            )
            fig, axes = plt.subplots(NROWS_2D_FLOW_FIELD, 1, figsize=FIGSIZE_2D_FLOW_FIELD)
            suptitle_str = f"{dataset_name} — Bootstrap-Validated Fixed Points\n(Detection Rate {Unicode.GEQ} {bootstrap_threshold:.2%}"
            suptitle_suffix = ")" if n_bootstrap is None else f", n bootstrap = {n_bootstrap})"
            fig.suptitle(f"{suptitle_str}{suptitle_suffix}")

            for ax, column_x, column_y in [
                (axes[0], column_names[0], column_names[1]),  # PC1 vs PC2
                (axes[1], column_names[0], column_names[2]),  # PC1 vs PC3
            ]:
                for _, row in high_confidence_df_flow.iterrows():
                    stability = row[Column.VectorField.STABILITY]
                    color = FIXED_POINT_PLOT_STYLE[stability].color
                    marker = FIXED_POINT_PLOT_STYLE[stability].marker

                    x = row[f"{column_x}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                    y = row[f"{column_y}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                    xlabel = get_label_for_column(column_x)
                    ylabel = get_label_for_column(column_y)

                    x_lo = row[f"{column_x}_{Column.BootstrapAnalysis.CI_LOWER}"]
                    x_hi = row[f"{column_x}_{Column.BootstrapAnalysis.CI_UPPER}"]
                    y_lo = row[f"{column_y}_{Column.BootstrapAnalysis.CI_LOWER}"]
                    y_hi = row[f"{column_y}_{Column.BootstrapAnalysis.CI_UPPER}"]
                    print(x_lo, x_hi, y_lo, y_hi)

                    xerr = (
                        [[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]]
                        if not (np.isnan(x_lo) or np.isnan(x_hi))
                        else None
                    )
                    yerr = (
                        [[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]]
                        if not (np.isnan(y_lo) or np.isnan(y_hi))
                        else None
                    )

                    ax.errorbar(
                        x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        fmt=marker,
                        color=color,
                        markeredgecolor="black",
                        markersize=8,
                        capsize=4,
                        elinewidth=1.2,
                        ecolor=color,
                        zorder=3,
                    )

                ax.set_xlim(bounds_for_plots[column_x])
                ax.set_ylim(bounds_for_plots[column_y])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{xlabel} vs {ylabel}")

            # Legend from the stability labels present in this dataset
            present_stabilities = set(
                high_confidence_df_flow[Column.VectorField.STABILITY].unique()
            )
            legend_handles = [
                StabilityLegendHandle(stability_label=s)
                for s in StabilityLabel
                if s in present_stabilities
            ]
            if legend_handles:
                axes[-1].legend(handles=legend_handles, title="Stability", loc="best")

            plt.tight_layout()

            save_plot_to_path(
                fig,
                fig_savedir,
                f"bootstrap_fixed_points_ci_{dataset_name}_shear_{flow_condition.shear_stress_bin}",
            )
            plt.close(fig)

    # Combined cross-dataset figure
    if len(all_high_confidence_dfs) < 2:
        logger.warning(
            "High-confidence fixed points found for fewer than two datasets; "
            "skipping combined comparison figure."
        )
        return

    combined_df = pd.concat(all_high_confidence_dfs, ignore_index=True)
    dataset_list = combined_df[Column.DATASET].unique().tolist()

    fig_combined, axes_combined = plt.subplots(
        NROWS_2D_FLOW_FIELD, 1, figsize=FIGSIZE_2D_FLOW_FIELD
    )
    suptitle_str = f"Bootstrap-Validated Stable Fixed Points — All Datasets\n(Detection Rate {Unicode.GEQ} {bootstrap_threshold:.2%}"
    fig_combined.suptitle(f"{suptitle_str}{suptitle_suffix}")

    for ax, column_x, column_y in [
        (axes_combined[0], column_names[0], column_names[1]),
        (axes_combined[1], column_names[0], column_names[2]),
    ]:
        for ds_name, ds_df in combined_df.groupby(Column.DATASET):
            ds_color = get_dataset_color(ds_name)
            for _, row in ds_df.iterrows():
                stability = row[Column.VectorField.STABILITY]
                # only plot stable fixed points in the combined figure for clearer comparison
                if stability != StabilityLabel.STABLE:
                    continue

                x = row[f"{column_x}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                y = row[f"{column_y}_{Column.BootstrapAnalysis.CLUSTER_MEAN}"]
                xlabel = get_label_for_column(column_x)
                ylabel = get_label_for_column(column_y)

                x_lo = row[f"{column_x}_{Column.BootstrapAnalysis.CI_LOWER}"]
                x_hi = row[f"{column_x}_{Column.BootstrapAnalysis.CI_UPPER}"]
                y_lo = row[f"{column_y}_{Column.BootstrapAnalysis.CI_LOWER}"]
                y_hi = row[f"{column_y}_{Column.BootstrapAnalysis.CI_UPPER}"]

                xerr = (
                    [[max(0.0, x - x_lo)], [max(0.0, x_hi - x)]]
                    if not (np.isnan(x_lo) or np.isnan(x_hi))
                    else None
                )
                yerr = (
                    [[max(0.0, y - y_lo)], [max(0.0, y_hi - y)]]
                    if not (np.isnan(y_lo) or np.isnan(y_hi))
                    else None
                )

                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt=marker,
                    color=ds_color,
                    markeredgecolor="black",
                    markersize=8,
                    capsize=4,
                    elinewidth=1.2,
                    ecolor=ds_color,
                    zorder=3,
                )

        ax.set_xlim(bounds_for_plots[column_x])
        ax.set_ylim(bounds_for_plots[column_y])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")

    dataset_legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=get_dataset_color(ds),
            linestyle="",
            markeredgecolor="black",
            markersize=8,
            label=ds,
        )
        for ds in dataset_list
    ]
    axes_combined[-1].legend(
        handles=dataset_legend_handles, title="Dataset", loc="best", fontsize=6
    )

    save_plot_to_path(fig_combined, fig_savedir, "bootstrap_stable_fixed_points_ci_combined")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
