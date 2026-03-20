from endo_pipeline.cli import Datasets
from endo_pipeline.settings.migration_coherence import DEFAULT_MIGRATION_COHERENCE_FEATURE


def main(
    datasets: Datasets | None = None,
    optical_flow_feature: str = DEFAULT_MIGRATION_COHERENCE_FEATURE,
    plot_fixed_points: bool = True,
) -> None:
    import logging

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        check_required_columns_in_dataframe,
        fit_pca,
        get_dataframe_for_dynamics_workflows,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_binned_mean_to_fixed_points,
        add_optical_flow_features,
        add_shear_stress_to_df,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.diffae_features.pplane import (
        make_legend_handles_for_fixed_pts,
    )
    from endo_pipeline.library.visualize.migration_coherence import (
        plot_3d_scatter_or_binned,
        plot_fixed_points_vs_shear_stress,
        plot_scatter_and_binned_heatmap,
    )
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
    from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLOR_DICT,
        STABILITY_COLUMN_NAME,
        STABILITY_MARKER_DICT,
    )
    from endo_pipeline.settings.migration_coherence import (
        MIGRATION_COHERENCE_CROP_PATTERN,
        MIGRATION_COHERENCE_HIST_BINWIDTH,
        MIGRATION_COHERENCE_HIST_FIGSIZE,
        MIGRATION_COHERENCE_HIST_NUM_BINS,
        MIGRATION_COHERENCE_HIST_PLOT_KDE,
    )
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # Load diffae features
    crop_pattern = MIGRATION_COHERENCE_CROP_PATTERN
    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    feature_dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern=crop_pattern
    )
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    # get fit PCA object to apply PCA transformation to diffae features before
    # plotting against optical flow features.
    pca = fit_pca(num_pcs=3)

    fixed_points_dataframe_manifest_name = (
        f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{feature_dataframe_manifest_name}"
    )
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    # If datasets aren't provided, default to processing a default list of
    # datasets
    dataset_names = datasets or get_datasets_in_collection(
        "diffae_model_training"
    ) + get_datasets_in_collection("replicate_2_datasets")

    # if in demo mode, only process the first dataset and log a warning
    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.warning(
            "Running in demo mode, only processing first dataset [ %s ]",
            dataset_names[0],
        )

    # initialize a single figure and axis for plotting the distribution of
    # optical flow features across datasets, which will be saved at the end of
    # the loop after plotting all datasets on the same axis
    fig_hist, ax_hist = plt.subplots(figsize=MIGRATION_COHERENCE_HIST_FIGSIZE)

    # Load optical flow features and plot against diffae features
    df_fp_all_list: list[pd.DataFrame] = []
    for dataset_name in dataset_names:
        if dataset_name not in feature_dataframe_manifest.locations:
            logger.warning(
                "No feature dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. Skipping this dataset.",
                dataset_name,
                feature_dataframe_manifest.name,
            )
            continue
        output_dir = get_output_path(__file__, dataset_name)

        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name,
            feature_dataframe_manifest,
            pca=pca,
            include_cell_piling=False,
            include_not_steady_state=False,
            crop_pattern=crop_pattern,
        )
        df_of = add_optical_flow_features(
            df_dataset,
            datasets=[dataset_name],
        )

        # split the dataframe by flow condition so we can plot the distribution
        # of optical flow features for each flow condition separately
        dataset_config = load_dataset_config(dataset_name)
        df_by_flow, shear_stress_list = split_dataset_by_flow(df_of, dataset_config)

        for df_flow, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
            dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
            plot_label = f"{dataset_name}, ({shear_stress} dyn/cm$^2$)"

            # add to running plot of optical flow feature distribution across
            # datasets by plotting the distribution for this dataset and flow
            # condition on the shared axis (ax_hist), using a different color
            # for each dataset and flow condition combination
            hist_color = get_dataset_color(dataset_name)
            sns.histplot(
                df_flow[optical_flow_feature],
                bins=MIGRATION_COHERENCE_HIST_NUM_BINS,
                kde=MIGRATION_COHERENCE_HIST_PLOT_KDE,
                label=plot_label,
                binwidth=MIGRATION_COHERENCE_HIST_BINWIDTH,
                ax=ax_hist,
                color=hist_color,
            )

            # initialize fixed_points_dataframe to None in case we aren't plotting
            # fixed points or if loading the fixed points dataframe fails for any
            # reason, then try to load the fixed points dataframe if we're plotting fixed points
            fixed_points_dataframe: pd.DataFrame | None = None
            if plot_fixed_points:
                try:
                    fixed_points_dataframe_location = get_dataframe_location_for_dataset(
                        fixed_points_dataframe_manifest, dataset_name
                    )
                    fixed_points_dataframe = load_dataframe(
                        fixed_points_dataframe_location, delay=False
                    )
                    check_required_columns_in_dataframe(
                        fixed_points_dataframe,
                        required_columns=[
                            *DYNAMICS_COLUMN_NAMES,
                            ColumnName.DATASET,
                            STABILITY_COLUMN_NAME,
                        ],
                    )
                except KeyError:
                    # if the fixed points dataframe for this dataset isn't found in
                    # the manifest, log a warning and continue without loading fixed
                    # points (i.e., fixed_points_dataframe will remain None)
                    logger.warning(
                        "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                        "Fixed points will not be overlaid on the migration coherence plots for this dataset.",
                        dataset_name,
                        fixed_points_dataframe_manifest.name,
                    )

            # --- 2D plots ---
            for x_col, y_col in [
                (ColumnName.POLAR_RADIUS, ColumnName.POLAR_ANGLE),
                (ColumnName.PC3_FLIPPED, ColumnName.POLAR_ANGLE),
                (ColumnName.POLAR_RADIUS, ColumnName.PC3_FLIPPED),
            ]:
                figure_filename = (
                    f"{dataset_name_flow}_{x_col}_vs_{y_col}_colored_by_{optical_flow_feature}"
                )
                logger.info(
                    "Plotting optical flow feature over [ %s ] vs [ %s ] for dataset [ %s ], shear stress [ %s ]",
                    x_col,
                    y_col,
                    dataset_name,
                    shear_stress,
                )
                fig, axs = plot_scatter_and_binned_heatmap(
                    df=df_flow,
                    x_col=x_col,
                    y_col=y_col,
                    color_col=optical_flow_feature,
                )
                plt.suptitle(plot_label)
                plt.tight_layout()
                save_plot_to_path(
                    fig,
                    output_dir,
                    figure_filename,
                )
                plt.close(fig)

                # if fixed points are available, overlay them on the scatter plot
                if fixed_points_dataframe is not None:
                    for _, row in fixed_points_dataframe.iterrows():
                        stability = row[STABILITY_COLUMN_NAME]
                        marker = STABILITY_MARKER_DICT.get(stability, "o")
                        color = STABILITY_COLOR_DICT.get(stability, "gray")
                        axs[1].scatter(
                            row[x_col],
                            row[y_col],
                            marker=marker,
                            color=color,
                            edgecolor="black",
                            s=100,
                            label=f"Fixed Point ({stability})",
                        )
                    # add legend for fixed points
                    legend_handles = make_legend_handles_for_fixed_pts(
                        fixed_points_dataframe[STABILITY_COLUMN_NAME].unique().tolist()
                    )
                    fig.legend(
                        handles=legend_handles,
                        bbox_to_anchor=(1.00, 0.90),
                        title="fixed point stability",
                        loc="upper left",
                        fontsize=10,
                    )
                    fig.tight_layout()
                    save_plot_to_path(
                        fig,
                        output_dir,
                        f"{figure_filename}_with_fixed_points",
                    )
                    plt.close(fig)

            # --- 3D plots ---
            df_flow_no_nan = df_flow.dropna(subset=[optical_flow_feature])

            if fixed_points_dataframe is not None:
                fixed_points_dataframe = add_binned_mean_to_fixed_points(
                    fixed_points_dataframe,
                    df_flow_no_nan,
                    x_col=ColumnName.POLAR_ANGLE,
                    y_col=ColumnName.POLAR_RADIUS,
                    z_col=ColumnName.PC3_FLIPPED,
                    binned_col=optical_flow_feature,
                )

            # 3D Scatter
            fig, ax = plot_3d_scatter_or_binned(
                df_flow_no_nan,
                x_col=ColumnName.POLAR_ANGLE,
                y_col=ColumnName.POLAR_RADIUS,
                z_col=ColumnName.PC3_FLIPPED,
                color_col=optical_flow_feature,
                df_fp=fixed_points_dataframe,
                binned=False,
            )
            ax.set_title(plot_label, loc="left")
            plt.show()
            save_plot_to_path(
                fig,
                output_dir,
                f"{dataset_name_flow}_3D_scatter_{optical_flow_feature}",
            )
            plt.close(fig)

            # 3D Binned Heatmap
            fig, ax = plot_3d_scatter_or_binned(
                df_flow_no_nan,
                x_col=ColumnName.POLAR_ANGLE,
                y_col=ColumnName.POLAR_RADIUS,
                z_col=ColumnName.PC3_FLIPPED,
                color_col=optical_flow_feature,
                df_fp=fixed_points_dataframe,
                binned=True,
            )
            ax.set_title(plot_label, loc="left")
            plt.show()
            save_plot_to_path(
                fig,
                output_dir,
                f"{dataset_name_flow}_3D_binned_heatmap_{optical_flow_feature}",
            )
            plt.close(fig)

            if fixed_points_dataframe is not None:
                df_fp_all_list.append(fixed_points_dataframe)

    # --- Cross-dataset: fixed point variables vs shear stress ---
    if df_fp_all_list:
        df_fp_all = pd.concat(df_fp_all_list, ignore_index=True)
        df_fp_all = add_shear_stress_to_df(df_fp_all)
        cross_dataset_output_dir = get_output_path(__file__)

        variables = [
            ColumnName.POLAR_ANGLE,
            ColumnName.POLAR_RADIUS,
            ColumnName.PC3_FLIPPED,
            f"mean_{optical_flow_feature}",
        ]
        labels = ["\u03b8", "r", "\u03c1", "migration coherence"]

        for var, label in zip(variables, labels, strict=False):
            ylim = (0, 1) if var == f"mean_{optical_flow_feature}" else None
            plot_fixed_points_vs_shear_stress(
                df_fp_all,
                var,
                label,
                output_dir=cross_dataset_output_dir,
                ylim=ylim,
            )

    # after plotting all datasets on the same axis, save the optical flow feature distribution plot
    ax_hist.set_xlabel(optical_flow_feature)
    ax_hist.set_ylabel("Count")
    ax_hist.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=8,
    )
    fig_hist.tight_layout()
    save_plot_to_path(
        fig_hist,
        get_output_path(__file__),
        f"{optical_flow_feature}_{'_'.join(dataset_names)}_distribution",
    )
    plt.close(fig_hist)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
