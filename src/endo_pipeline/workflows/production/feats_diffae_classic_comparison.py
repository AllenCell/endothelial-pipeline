from endo_pipeline.cli import Datasets, TrackIds
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)


def main(
    datasets: Datasets | None = None,
    positions: list[int] | None = None,
    track_ids: TrackIds | None = None,
    datasets_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    track_integrations_only: bool = False,
    use_global_pc_lims: bool = False,
    for_figures: bool = False,
    n_cores: int = 30,
) -> None:
    """This workflow outputs plots of track-based cell trajectories integrated with grid-based DiffAE flow fields."""

    from collections import namedtuple
    from multiprocessing import Pool
    from typing import Literal

    from matplotlib import pyplot as plt
    from tqdm import tqdm

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, save_plot_to_path
    from endo_pipeline.library.analyze.integration.track_integration import (
        get_gridcrop_and_cellcentric_trajectories_and_flow_fields,
        get_preprocessed_manifests_and_km_bounds,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import (
        PlotMeasFeatAndFlowFieldOverlayArgs,
        multiproc_plot_measured_feat_overlay_on_flowfield,
        overlay_trajectory_heatmap_on_flowfield,
        plot_measured_feat_overlay_on_flowfield,
        plot_new_traj_overlay_on_grid_traj_and_flowfield,
        plot_quiver_slices_from_diffae_table,
    )
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings import NUM_PCS_TO_ANALYZE

    out_dir = get_output_path(__file__)
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    model_manifest = load_model_manifest(model_manifest_name)

    # create subdirectory to save track-based trajectories to
    out_subdir_traj = out_dir / "trajectories_track_based"
    out_subdir_traj.mkdir(parents=True, exist_ok=True)

    positions = list(range(6)) if positions is None else positions

    for dataset_name in datasets:

        # load and preprocess the different diffae manifests and PCA pipeline
        df_all_positions, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
            dataset_name=dataset_name,
            model_manifest=model_manifest,
            run_name=run_name,
            seg_feature_manifest_name=seg_feature_manifest_name,
            collection_name_for_pca=datasets_for_pca,
            num_pcs=NUM_PCS_TO_ANALYZE,
            drop_rows_without_diffae_feats=True,
            filtered=True,
        )

        # load or compute the trajectories and flow fields for the grid-based
        # and cell-centric crops
        traj_grids, flow_field_dict_grids, traj_tracks, _ = (
            get_gridcrop_and_cellcentric_trajectories_and_flow_fields(
                dataset_name=dataset_name,
                merged_feats_df=df_all_positions,
                diffae_grid_crops=diffae_grid_crops,
                bounds=bounds,
                trajectory_dir=out_subdir_traj,
            )
        )

        # save plots of the track-based crop trajectories and PCs overlaid
        # on the flow field and trajectories from the grid-based crops
        figure_format: Literal[".png", ".svg", ".pdf"] = ".pdf" if for_figures else ".png"

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

        if not track_integrations_only:
            # plot just the flow field (used for validation with the workflow
            # generate_3d_flow_field.py that this one is based on)
            fig, axs = plot_quiver_slices_from_diffae_table(
                diffae_grid_crops,
                traj_grids,
                flow_field_dict_grids,
                plot_trajectory=False,
                plot_fixed_points=True,
            )
            plt.tight_layout()
            save_plot_to_path(
                figure=fig,
                output_path=out_subdir,
                figure_name=f"{dataset_name}_flow_field",
                file_format=figure_format,
            )
            plt.close(fig)

            # plot the flow field and the trajectories for the average behavior of
            # tracked cells over time
            plot_new_traj_overlay_on_grid_traj_and_flowfield(
                out_subdir,
                dataset_name,
                diffae_grid_crops,
                traj_grids,
                flow_field_dict_grids,
                traj_tracks,
                figure_format=figure_format,
                use_global_pc_lims=use_global_pc_lims,
            )

            time_start = df_all_positions["time_hours"].min()
            time_stop = df_all_positions["time_hours"].max()

            FeatureLimitsPair = namedtuple(
                "FeatureLimitsPair", ["feature_name", "feature_hue_limits"]
            )
            measured_feats_to_plot = (
                FeatureLimitsPair("time_hours", (time_start, time_stop)),
                FeatureLimitsPair("alignment_deg_rel_to_flow", (0, 90)),
                FeatureLimitsPair("eccentricity", (0.0, 1.0)),
            )
            for feature_name, feature_hue_lims in measured_feats_to_plot:
                plot_measured_feat_overlay_on_flowfield(
                    out_subdir,
                    dataset_name,
                    diffae_grid_crops,
                    traj_grids,
                    flow_field_dict_grids,
                    diffae_measured_feat_df=df_all_positions,
                    meas_feat_col_name_for_color_coding=feature_name,
                    track_id_to_plot="mean",
                    plot_trajectory=False,
                    plot_fixed_points=True,
                    indicate_track_start=False,
                    indicate_track_end=True,
                    hue_norm=feature_hue_lims,
                    legend=legend,
                    alpha=0.8,
                    show_plot=False,
                    figure_format=figure_format,
                    use_global_pc_lims=use_global_pc_lims,
                )

            # plot trajectory heatmap (indicates where most of the data is in PC-space)
            overlay_trajectory_heatmap_on_flowfield(
                out_dir=out_subdir_heatmap,
                dataset_name=dataset_name,
                diffae_grid_crops=diffae_grid_crops,
                traj_grids=traj_grids,
                flow_field_dict_grids=flow_field_dict_grids,
                df_all_positions=df_all_positions,
            )

        # plot single track examples
        for pos in positions:
            df_one_position = df_all_positions.query("position == @pos")
            # for pos, df_one_position in df_all_positions.groupby("position_as_str"):
            out_subdir_indiv_pos = out_subdir_indiv / str(pos)
            out_subdir_indiv_pos.mkdir(parents=True, exist_ok=True)

            # decide on the feature to color code by and the min and max
            # of that color coding
            measured_feature = "alignment_deg_rel_to_flow"
            hue_norm = (0, 90)

            if track_ids is None:
                track_ids = sorted(df_one_position["track_id"].unique().tolist())
                # only overlay every 10th track id if there are a lot
                # of tracks to save time + space
                track_ids = track_ids[::10] if len(track_ids[::10]) > 10 else track_ids

            arg_list = []
            for tid in track_ids:
                arg_list.append(
                    PlotMeasFeatAndFlowFieldOverlayArgs(
                        out_subdir_indiv_pos,
                        dataset_name,
                        diffae_grid_crops,
                        traj_grids,
                        flow_field_dict_grids,
                        df_one_position,
                        measured_feature,
                        tid,
                        hue_norm,
                        legend,
                        figure_format,
                        use_global_pc_lims,
                    )
                )

            # make the plots
            with Pool(processes=n_cores) as pool:
                print(f"Starting multiprocessing pool for plotting (using {n_cores} cores)...")
                list(
                    tqdm(
                        pool.imap(
                            multiproc_plot_measured_feat_overlay_on_flowfield, arg_list, chunksize=5
                        ),
                        total=len(arg_list),
                        desc=f"Plotting tracks at {dataset_name} {pos}",
                    )
                )
                pool.close()
                pool.join()
                print("Multiprocessing pool for plotting complete.")


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
