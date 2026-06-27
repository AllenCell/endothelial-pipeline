from typing import Annotated, Literal

from cyclopts import Parameter


def main(
    dataset_group: Literal["low_high", "intermediate", "perturbation"] = "intermediate",
    include_fixed_points: Annotated[bool, Parameter(negative="--exclude-fixed-points")] = True,
    skip_individual_plots: Annotated[bool, Parameter(negative="--make-individual-plots")] = False,
) -> None:
    """
    Visualize migration coherence relative to fixed points.

    #migration-coherence #fixed-points

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe plot-migration-coherence -vd
    ```

    To run the workflow for a specific dataset group:

    ```bash
    uv run endopipe plot-migration-coherence --dataset-group DATASET_GROUP
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will plot
    migration coherence for the first dataset in the group.

    Parameters
    ----------
    dataset_group
        Group of datasets to include in analysis.
    include_fixed_points
        True to overlay fixed points on the plots, False otherwise.
    skip_individual_plots
        True to skip generating individual plots for each dataset and flow
        condition, False otherwise.
    """

    import logging

    import matplotlib.pyplot as plt
    import pandas as pd

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import (
        TimepointAnnotation,
        get_shear_stress_label_for_dataset,
        load_dataset_config,
    )
    from endo_pipeline.io import (
        get_output_path,
        join_sorted_strings,
        load_dataframe,
        save_plot_to_path,
    )
    from endo_pipeline.library.analyze.dataframe_filtering import (
        filter_dataframe_by_annotations,
        filter_dataframe_to_flow_condition_by_timepoint,
    )
    from endo_pipeline.library.analyze.dataframe_validation import (
        check_required_columns_in_dataframe,
    )
    from endo_pipeline.library.analyze.migration_coherence.optical_flow_feature import (
        add_binned_mean_to_fixed_points,
        add_optical_flow_features,
    )
    from endo_pipeline.library.visualize.diffae_features.dynamics import (
        make_legend_handles_for_fixed_pts,
    )
    from endo_pipeline.library.visualize.diffae_features.feature_viz import get_dataset_color
    from endo_pipeline.library.visualize.migration_coherence import (
        plot_3d_scatter_or_binned,
        plot_optical_flow_histogram,
        plot_scatter_and_binned_heatmap,
    )
    from endo_pipeline.library.visualize.summary_plot import (
        build_dataframe_for_fixed_point_dataset_summary,
        plot_cross_dataset_summaries,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName
    from endo_pipeline.settings.column_names import ColumnNameTemplate as ColumnTemplate
    from endo_pipeline.settings.dynamics_workflows import (
        DYNAMICS_COLUMN_NAMES,
        METADATA_COLUMNS_TO_KEEP,
    )
    from endo_pipeline.settings.manifest_names import (
        BOOTSTRAPPING_MANIFEST_NAMES,
        DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    )
    from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_PATCH_TYPE
    from endo_pipeline.settings.plot_defaults import FIXED_POINT_PLOT_STYLE
    from endo_pipeline.settings.summary_plot import SUMMARY_PLOT_DATASETS
    from endo_pipeline.settings.unicode import UnicodeCharacters as Unicode
    from endo_pipeline.settings.workflow_defaults import FEATURES_FILTERED_MANIFEST_NAMES

    logger = logging.getLogger(__name__)

    # Load diffae features
    feature_dataframe_manifest_name = FEATURES_FILTERED_MANIFEST_NAMES[
        MIGRATION_COHERENCE_PATCH_TYPE
    ]
    feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

    feature_column_names = list(DYNAMICS_COLUMN_NAMES)
    fp_column_names = [ColumnTemplate.FIXED_POINT % col for col in DYNAMICS_COLUMN_NAMES]
    columns_to_compute = [*METADATA_COLUMNS_TO_KEEP["grid_based"], *feature_column_names]

    name_suffix = f"_{join_sorted_strings(feature_column_names)}_{MIGRATION_COHERENCE_PATCH_TYPE}"
    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}{name_suffix}"
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    bootstrap_manifest_name = BOOTSTRAPPING_MANIFEST_NAMES[MIGRATION_COHERENCE_PATCH_TYPE]
    fixed_points_bootstrap_dataframe_manifest = load_dataframe_manifest(bootstrap_manifest_name)

    output_dir = get_output_path(__file__, dataset_group)

    datasets = SUMMARY_PLOT_DATASETS[dataset_group]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset")
        datasets = datasets[:1]

    # --- Cross-dataset summary plots ---
    dataset_summary_df = build_dataframe_for_fixed_point_dataset_summary(
        dataset_names=datasets,
        feature_dataframe_manifest=feature_dataframe_manifest,
        bootstrap_dataframe_manifest=fixed_points_bootstrap_dataframe_manifest,
        convert_angle_to_nematic=True,
        stable_only=True,
    )
    plot_cross_dataset_summaries(
        dataset_summary_df,
        output_path=output_dir,
        axis_mode="dataset",
        category_order=datasets,
        subplot_layout="vertical",
        ylabel_rotation=90,
    )

    if skip_individual_plots:
        return

    for optical_flow_feature, vmax, hist_binwidth in [
        (ColumnName.OpticalFlow.UNIT_VECTOR_MEAN, 1, 0.02),
        (ColumnName.OpticalFlow.SPEED_MEAN, 10, 0.2),
    ]:
        for dataset_name in datasets:
            if dataset_name not in feature_dataframe_manifest.locations:
                logger.warning(
                    "Dataset '%s' not found in manifest '%s'. Skipping.",
                    dataset_name,
                    feature_dataframe_manifest,
                )
                continue

            output_dir = get_output_path(__file__, optical_flow_feature, dataset_name)
            dataset_config = load_dataset_config(dataset_name)

            # load dataframe and perform additional filtering (remove
            # non-steady-state timepoints based on annotations), computing
            # only the columns needed for visualization/analysis
            df = load_dataframe(feature_dataframe_manifest.locations[dataset_name], delay=True)
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
            for flow_condition in dataset_config.flow_conditions:
                dataset_name_flow = f"{dataset_name}_shear_{flow_condition.shear_stress_bin}"
                df_flow = filter_dataframe_to_flow_condition_by_timepoint(
                    df_of, dataset_config, flow_condition
                )
                plot_label = get_shear_stress_label_for_dataset(dataset_config, flow_condition)
                hist_color = get_dataset_color(dataset_name)

                # load fixed points once per dataset
                fixed_points_dataframe: pd.DataFrame | None = None
                if include_fixed_points:
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
                                *fp_column_names,
                                ColumnName.DATASET,
                                ColumnName.FIXED_POINT_STABILITY,
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
                        fp_x_col=ColumnTemplate.FIXED_POINT % ColumnName.DiffAEData.POLAR_ANGLE,
                        fp_y_col=ColumnTemplate.FIXED_POINT % ColumnName.DiffAEData.POLAR_RADIUS,
                        fp_z_col=ColumnTemplate.FIXED_POINT % ColumnName.DiffAEData.PC3_FLIPPED,
                        of_x_col=ColumnName.DiffAEData.POLAR_ANGLE,
                        of_y_col=ColumnName.DiffAEData.POLAR_RADIUS,
                        of_z_col=ColumnName.DiffAEData.PC3_FLIPPED,
                        binned_col=optical_flow_feature,
                    )

                # save individual histogram for this dataset and flow condition
                fig = plot_optical_flow_histogram(
                    df=df_flow,
                    optical_flow_feature=optical_flow_feature,
                    feature_label="Migration Coherence",
                    feature_lim=(0.1, vmax),
                    ss_label=f"{int(flow_condition.shear_stress)} dyn/cm{Unicode.SQUARED}",
                    color=hist_color,
                    df_fp=fp_for_feature,
                    binwidth=hist_binwidth,
                )
                save_plot_to_path(
                    fig, output_dir, f"{dataset_name_flow}_{optical_flow_feature}_distribution"
                )

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
                            stability = row[ColumnName.FIXED_POINT_STABILITY]
                            marker = FIXED_POINT_PLOT_STYLE[stability].marker
                            color = FIXED_POINT_PLOT_STYLE[stability].color
                            axs[1].scatter(
                                row[ColumnTemplate.FIXED_POINT % x_col],
                                row[ColumnTemplate.FIXED_POINT % y_col],
                                marker=marker,
                                color=color,
                                edgecolor="black",
                                s=100,
                                label=f"Fixed Point ({stability})",
                            )
                        # add legend for fixed points
                        legend_handles = make_legend_handles_for_fixed_pts(
                            fixed_points_dataframe[ColumnName.FIXED_POINT_STABILITY]
                            .unique()
                            .tolist()
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
                    fp_template=ColumnTemplate.FIXED_POINT,
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
                    fp_template=ColumnTemplate.FIXED_POINT,
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
