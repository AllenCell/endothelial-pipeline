from endo_pipeline.cli import Datasets


def main(
    datasets: Datasets | None = None,
    optical_flow_feature: str = "optical_flow_mean_unit_vector_dt1",
    plot_fixed_points: bool = True,
    skip_individual_plots: bool = False,
) -> None:
    """
    Analyze the coherence of migration in relation to fixed points identified in the structure
    feature space.

    datasets:
        Optional list of dataset names to include in the analysis.
        If not provided, defaults to all datasets in the "optical_flow_analysis" collection.
        To apply to the KD datasets, set to "perturbation".
    optical_flow_feature:
        The optical flow feature to analyze. This should be a column in the feature dataframes.
        To analyze the speed, use "optical_flow_mean_speed_dt1".
    plot_fixed_points:
        Whether to overlay fixed points on the migration coherence plots.
    skip_individual_plots:
        Whether to skip generating individual plots for each dataset and flow condition.
        If True, only the cross-dataset summary plots will be generated.
    """
    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_datasets_in_collection,
        load_dataset_config,
    )
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        filter_dataframe_by_annotations,
        split_dataset_by_flow,
    )
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_binned_mean_to_fixed_points,
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.diffae_features.pplane import (
        make_legend_handles_for_fixed_pts,
    )
    from endo_pipeline.library.visualize.migration_coherence import (
        plot_3d_scatter_or_binned,
        plot_cross_dataset_summaries,
        plot_optical_flow_histogram,
        plot_scatter_and_binned_heatmap,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.flow_field_dataframes import (
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
        STABILITY_COLOR_DICT,
        STABILITY_COLUMN_NAME,
        STABILITY_MARKER_DICT,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    # Load diffae features
    base_name = (
        f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{MIGRATION_COHERENCE_CROP_PATTERN}"
    )
    feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    # If datasets aren't provided, default to processing a default list of datasets
    dataset_names = datasets or get_datasets_in_collection("optical_flow_analysis")
    if datasets is None:
        collection = "optical_flow_analysis"
    else:
        collection = "perturbation"

    output_dir = get_output_path(__file__, collection, optical_flow_feature)

    if DEMO_MODE:
        dataset_names = dataset_names[:1]
        logger.info("DEMO MODE, only processing first dataset [ %s ]", dataset_names[0])

    # --- Cross-dataset summary plots ---
    plot_cross_dataset_summaries(
        dataset_names=dataset_names,
        optical_flow_feature=optical_flow_feature,
        feature_dataframe_manifest=feature_dataframe_manifest,
        fixed_points_dataframe_manifest=fixed_points_dataframe_manifest,
        output_dir=output_dir,
        plot_fixed_points=plot_fixed_points,
        by_dataset=True,
    )

    if not skip_individual_plots:
        for dataset_name in dataset_names:
            if dataset_name not in feature_dataframe_manifest.locations:
                logger.warning(
                    "No feature dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. Skipping this dataset.",
                    dataset_name,
                    feature_dataframe_manifest.name,
                )
                continue
            output_dir = get_output_path(__file__, collection, optical_flow_feature, dataset_name)
            dataset_config = load_dataset_config(dataset_name)

            # load dataframe and perform additional filtering (remove
            # non-steady-state timepoints based on annotations), computing
            # only the columns needed for visualization/analysis
            df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
            columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid"], *DYNAMICS_COLUMN_NAMES]
            df_ = df[columns_to_compute].compute()
            df_steady_state = filter_dataframe_by_annotations(
                df_,
                dataset_config,
                timepoint_annotations=[TimepointAnnotation.NOT_STEADY_STATE],
            )

            df_of = add_optical_flow_features(
                df_steady_state,
                datasets=[dataset_name],
            )

            # split the dataframe by flow condition so we can plot the distribution
            # of optical flow features for each flow condition separately
            df_by_flow, shear_stress_list = split_dataset_by_flow(df_of, dataset_config)

            for df_flow, shear_stress in zip(df_by_flow, shear_stress_list, strict=True):
                dataset_name_flow = f"{dataset_name}_shear_{int(shear_stress)}"
                plot_label = f"{dataset_name} ({int(shear_stress)} dyn/cm$^2$)"
                hist_color = get_dataset_color(dataset_name)

                # load fixed points once per dataset
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
                        logger.warning(
                            "No fixed point dataframe found for dataset [ %s ] in dataframe manifest [ %s ]. "
                            "Fixed points will not be overlaid on the migration coherence plots for this dataset.",
                            dataset_name,
                            fixed_points_dataframe_manifest.name,
                        )

                # Enrich fixed points with binned mean of the optical flow
                # feature so downstream plots (histogram, 3D) can use it.
                df_flow_no_nan = df_flow.dropna(subset=[optical_flow_feature])
                fp_for_feature = fixed_points_dataframe
                if fp_for_feature is not None:
                    fp_for_feature = add_binned_mean_to_fixed_points(
                        fp_for_feature,
                        df_flow_no_nan,
                        x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                        y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                        z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                        binned_col=optical_flow_feature,
                    )

                # save individual histogram for this dataset and flow condition
                plot_optical_flow_histogram(
                    df=df_flow,
                    optical_flow_feature=optical_flow_feature,
                    title=plot_label,
                    color=hist_color,
                    output_dir=output_dir,
                    filename=f"{dataset_name_flow}_{optical_flow_feature}_distribution",
                    df_fp=fp_for_feature,
                )

                if "unit_vector" in optical_flow_feature:
                    vmax = 1
                if "speed" in optical_flow_feature:
                    vmax = 10

                # --- 2D plots ---
                for x_col, y_col in [
                    (ColumnName.DiffAEData.POLAR_RADIUS, ColumnName.DiffAEData.POLAR_ANGLE),
                    (ColumnName.DiffAEData.PC3_FLIPPED, ColumnName.DiffAEData.POLAR_ANGLE),
                    (ColumnName.DiffAEData.POLAR_RADIUS, ColumnName.DiffAEData.PC3_FLIPPED),
                ]:
                    figure_filename = (
                        f"{dataset_name_flow}_{x_col}_vs_{y_col}_colored_by_{optical_flow_feature}"
                    )
                    fig, axs = plot_scatter_and_binned_heatmap(
                        df=df_flow,
                        x_col=x_col,
                        y_col=y_col,
                        vmin=0,
                        vmax=vmax,
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
                # 3D Scatter
                fig, ax = plot_3d_scatter_or_binned(
                    df_flow_no_nan,
                    x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                    y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                    z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                    color_col=optical_flow_feature,
                    df_fp=fp_for_feature,
                    binned=False,
                    vmax=vmax,
                )
                ax.set_title(plot_label, loc="left")
                save_plot_to_path(
                    fig,
                    output_dir,
                    f"{dataset_name_flow}_3D_scatter_{optical_flow_feature}",
                )
                plt.close(fig)

                # 3D Binned Heatmap
                fig, ax = plot_3d_scatter_or_binned(
                    df_flow_no_nan,
                    x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                    y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                    z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                    color_col=optical_flow_feature,
                    df_fp=fp_for_feature,
                    binned=True,
                    vmax=vmax,
                )
                ax.set_title(plot_label, loc="left")
                save_plot_to_path(
                    fig,
                    output_dir,
                    f"{dataset_name_flow}_3D_binned_heatmap_{optical_flow_feature}",
                )
                plt.close(fig)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
