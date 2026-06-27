from endo_pipeline.cli import Datasets, UniqueIntList


def main(
    datasets: Datasets | None = None,
    positions: UniqueIntList | None = None,
    track_ids_to_overlay: UniqueIntList | None = None,
    make_trajectory_summary_plots: bool = True,
    use_global_pc_lims: bool = False,
    num_processes: int = 1,
) -> None:
    """
    Compare cell-centered trajectories on grid-based flow fields.

    #grid-based #cell-centered

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe compare-flow-field-trajectories -vd
    ```

    To run the workflow for a single dataset:

    ```bash
    uv run endopipe compare-flow-field-trajectories --datasets DATASET_NAME
    ```

    ## Dataset collection

    If datasets are not provided, the workflow will use datasets in the
    `diffae_model_training` dataset collection.

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will run the
    comparison on a single position of a single dataset for 10 tracks.

    Parameters
    ----------
    datasets
        List of datasets or dataset collections to compare.
    positions
        List of positions to compare.
    track_ids_to_overlay
        Specific track IDs to overlay on flow fields.
    make_trajectory_summary_plots
        True to plot trajectory summaries, False otherwise.
    use_global_pc_lims
        True to use same PC limits for all datasets, False to set limit
        individually for each dataset.
    num_processes
        Number of processes to use.
    """

    import logging
    from collections import namedtuple
    from typing import Literal

    from matplotlib import pyplot as plt

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
    from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
    from endo_pipeline.library.analyze.track_integration import (
        get_flow_field_and_fixed_points,
        get_flow_field_estimation_bin_widths,
    )
    from endo_pipeline.library.process.general_image_preprocessing import process_task_queue
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        PlotMeasFeatAndFlowFieldOverlayArgs,
        multiproc_plot_measured_feat_overlay_on_flowfield,
        overlay_feature_on_flowfield,
        overlay_trajectory_heatmap_on_flowfield,
        plot_quiver_slices_from_flow_field_dict,
        save_feature_flowfield_overlay,
    )
    from endo_pipeline.manifests import get_dataframe_location_for_dataset
    from endo_pipeline.manifests.dataframe_manifest_io import load_dataframe_manifest
    from endo_pipeline.settings.column_names import ColumnName as Column
    from endo_pipeline.settings.dynamics_workflows import (
        DEFAULT_DATASETS_DYNAMICS_VIS,
        DYNAMICS_COLUMN_NAMES,
        LONG_TRACK_THRESHOLD_LENGTH,
    )
    from endo_pipeline.settings.workflow_defaults import (
        CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME,
        DATASET_INFO_COLUMNS,
        DEFAULT_MODEL_MANIFEST_NAME,
        DEFAULT_MODEL_RUN_NAME,
    )

    logger = logging.getLogger(__name__)

    out_dir = get_output_path(__file__)

    dataset_names = datasets or get_datasets_in_collection(DEFAULT_DATASETS_DYNAMICS_VIS)

    dynamics_columns = list(DYNAMICS_COLUMN_NAMES)
    seg_feat_columns = [
        Column.SegData.TIME_HRS,
        Column.SegData.ALIGNMENT_DEG,
        Column.SegData.ORIENTATION_DEG,
        Column.SegData.ECCENTRICITY,
    ]

    if DEMO_MODE:
        logger.warning("DEMO MODE - Limiting to one dataset, one position, and 10 tracks")
        dataset_names = dataset_names[:1]
        max_positions = 1
        max_tracks = 10
    else:
        max_positions = None
        max_tracks = None

    for dataset_name in dataset_names:
        dataset_config = load_dataset_config(dataset_name)

        positions = positions or dataset_config.zarr_positions
        if max_positions is not None:
            positions = positions[:max_positions]

        # define the columns to compute
        compute_cols = [
            *DATASET_INFO_COLUMNS,
            *dynamics_columns,
            *seg_feat_columns,
            Column.TRACK_LENGTH,
        ]

        # load and preprocess the different diffae manifests and PCA pipeline
        cellcentric_manifest = load_dataframe_manifest(
            CELL_CENTERED_FEATURES_FILTERED_MANIFEST_NAME
        )
        cellcentric_df_location = get_dataframe_location_for_dataset(
            cellcentric_manifest, dataset_name
        )
        cellcentric_df_delayed = load_dataframe(cellcentric_df_location, delay=True)
        cellcentric_df = (
            cellcentric_df_delayed[list(set(compute_cols) & set(cellcentric_df_delayed.columns))]
            .compute()
            .reset_index(drop=True)
        )

        flow_field_dict_grids, fixed_points_df = get_flow_field_and_fixed_points(
            dataset_name=dataset_name,
            column_names=dynamics_columns,
            model_manifest_name=DEFAULT_MODEL_MANIFEST_NAME,
            run_name=DEFAULT_MODEL_RUN_NAME,
        )

        # Define some initial parameters for plotting
        # save plots of the track-based crop trajectories and PCs overlaid
        # on the flow field and trajectories from the grid-based crops
        figure_format: Literal[".png", ".svg", ".pdf"] = ".png"

        # create a subdirectory to save the plots to
        out_subdir = out_dir / dataset_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        # create subdirectory to save individual track overlay examples to
        out_subdir_indiv = out_subdir / "individual_track_overlays"
        out_subdir_indiv.mkdir(parents=True, exist_ok=True)

        # create subdirectory to save trajectory heatmap plots to
        out_subdir_heatmap = out_subdir / "trajectory_heatmap"
        out_subdir_heatmap.mkdir(parents=True, exist_ok=True)

        # first decide if the colormap should be shown in the plot as a legend
        legend: Literal["auto", "brief", "full", False] = "auto"

        # get some parameters for plotting integrated measured features over the flow field
        time_start = 0  # cellcentric_df[Column.SegData.TIME_HRS].min()
        time_stop = 48  # cellcentric_df[Column.SegData.TIME_HRS].max()

        FeatureLimitsPair = namedtuple("FeatureLimitsPair", ["feature_name", "feature_hue_limits"])
        measured_feats_to_plot = (
            FeatureLimitsPair(Column.SegData.TIME_HRS, (time_start, time_stop)),
            FeatureLimitsPair(Column.SegData.ALIGNMENT_DEG, (0, 90)),
            FeatureLimitsPair(Column.SegData.ECCENTRICITY, (0.0, 1.0)),
        )

        # Make trajectory summary plots, if desired
        if make_trajectory_summary_plots:
            # slice the flow field at fixed points and plot those slices
            for i, fp_row in fixed_points_df.iterrows():
                flow_field_slices = (
                    fp_row[dynamics_columns[2]],
                    fp_row[dynamics_columns[1]],
                )  # feature 3, feature 2
                fixed_points_at_slices = (
                    fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[2]]),
                    fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[1]]),
                )

                # plot just the flow field (used for validation with the workflow
                # visualize_3d_flow_field.py that this one is based on)
                # plot base flow field
                fig, axs = plot_quiver_slices_from_flow_field_dict(
                    dataset_name=dataset_name,
                    flow_field_dict_grids=flow_field_dict_grids,
                    feature_vals=flow_field_slices,
                    column_names=dynamics_columns,
                )
                for j, ax in enumerate(axs):
                    ax.scatter(*fixed_points_at_slices[j], c="k", s=50)
                plt.tight_layout()
                save_plot_to_path(
                    figure=fig,
                    output_path=out_subdir,
                    figure_name=f"{dataset_name}_flow_field_fp{i}",
                    file_format=figure_format,
                )
                plt.close(fig)

                # plot the flow field and the mean behavior of the tracked cells over time
                for feature_name, feature_hue_lims in measured_feats_to_plot:
                    fig, axs = plot_quiver_slices_from_flow_field_dict(
                        dataset_name=dataset_name,
                        flow_field_dict_grids=flow_field_dict_grids,
                        feature_vals=flow_field_slices,
                        column_names=dynamics_columns,
                    )
                    for j, ax in enumerate(axs):
                        ax.scatter(*fixed_points_at_slices[j], c="k", s=50)
                    fig, axs = overlay_feature_on_flowfield(
                        flowfield_fig_and_axs=(fig, axs),
                        cellcentric_df=cellcentric_df,
                        column_names=dynamics_columns,
                        column_name_for_color_coding=feature_name,
                        indicate_track_start=False,
                        indicate_track_end=True,
                        track_id_to_plot="mean",
                        hue_norm=feature_hue_lims,
                        legend=legend,
                        alpha=0.8,
                        use_global_pc_lims=use_global_pc_lims,
                    )
                    save_feature_flowfield_overlay(
                        out_dir=out_subdir,
                        flow_field_figure=fig,
                        dataset_name=dataset_name,
                        column_name_for_color_coding=feature_name,
                        track_id_to_plot="mean",
                        show_plot=False,
                        figure_format=figure_format,
                    )

                # plot trajectory heatmap (indicates where most of the data is in PC-space)
                bin_widths = get_flow_field_estimation_bin_widths(column_names=dynamics_columns)
                overlay_trajectory_heatmap_on_flowfield(
                    out_dir=out_subdir_heatmap,
                    dataset_name=dataset_name,
                    flow_field_dict_grids=flow_field_dict_grids,
                    feature_vals=flow_field_slices,
                    dynamics_columns=dynamics_columns,
                    cellcentric_df=cellcentric_df,
                    bin_widths=bin_widths,
                )

        # plot single track examples
        for pos in positions:
            df_one_position = cellcentric_df[cellcentric_df[Column.POSITION] == pos]

            df_one_position = df_one_position[
                df_one_position[Column.TRACK_LENGTH] > LONG_TRACK_THRESHOLD_LENGTH
            ]

            if df_one_position.empty:
                logger.warning(
                    f"No tracks longer than {LONG_TRACK_THRESHOLD_LENGTH} for position {pos}, skipping..."
                )
                continue

            # decide on the feature to color code by and the min and max
            # of that color coding
            measured_feature = Column.SegData.ALIGNMENT_DEG
            hue_norm = (0, 90)

            if track_ids_to_overlay is None:
                track_ids = sorted(df_one_position[Column.TRACK_ID].unique().tolist())
                # overlay every 10th track ID if there are a lot of tracks to save time + space
                track_ids = track_ids[::10] if len(track_ids[::10]) > 10 else track_ids
            else:
                track_ids = track_ids_to_overlay

            if max_tracks is not None:
                track_ids = track_ids[:max_tracks]

            for i, fp_row in fixed_points_df.iterrows():

                out_subdir_indiv_pos_one_fixedpoint = (
                    out_subdir_indiv / str(pos) / f"fixed_point_{i}"
                )
                out_subdir_indiv_pos_one_fixedpoint.mkdir(parents=True, exist_ok=True)

                flow_field_slices = (
                    fp_row[dynamics_columns[2]],
                    fp_row[dynamics_columns[1]],
                )  # feature 3, feature 2
                fixed_points_at_slices = (
                    fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[2]]),
                    fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[1]]),
                )

                arg_list = []
                for tid in track_ids:
                    arg_list.append(
                        PlotMeasFeatAndFlowFieldOverlayArgs(
                            out_subdir_single_position=out_subdir_indiv_pos_one_fixedpoint,
                            dataset_name=dataset_name,
                            flow_field_dict_grids=flow_field_dict_grids,
                            flow_field_slices=flow_field_slices,
                            fixed_points_at_slices=fixed_points_at_slices,
                            df_one_position=df_one_position,
                            measured_feature=measured_feature,
                            dynamics_columns=dynamics_columns,
                            track_id=tid,
                            hue_norm=hue_norm,
                            legend=legend,
                            figure_format=figure_format,
                            use_global_pc_lims=use_global_pc_lims,
                        )
                    )

                # make the plots
                logger.info(
                    f"Plotting track overlays for fixed point {i} (using {num_processes} cores)..."
                )
                process_task_queue(
                    task=multiproc_plot_measured_feat_overlay_on_flowfield,
                    queue=arg_list,
                    description=f"Plotting tracks at {dataset_name} P{pos} - fixed point {i}",
                    num_processes=num_processes,
                    chunksize=5,
                )
                logger.info("Plotting track overlays complete.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
