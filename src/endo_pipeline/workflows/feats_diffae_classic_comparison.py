import logging
from pathlib import Path

from src.endo_pipeline.configs import get_datasets_in_collection
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_features.track_integration import (
    add_normalized_time,
    get_gridcrop_and_cellcentric_trajectories_and_flow_fields,
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import make_all_plots

logger = logging.getLogger(__name__)


def main() -> None:
    out_dir = get_output_path(Path(__file__).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name_list = get_datasets_in_collection("pca_reference")

    # create subdirectory to save track-based trajectories to
    out_subdir_traj = out_dir / "trajectories_track_based"
    out_subdir_traj.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_name_list:

        # load and preprocess the different diffae manifests and PCA pipeline
        df_all_positions, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
            dataset_name
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

        # add normalized time to tracks
        df_all_positions = add_normalized_time(df_all_positions)

        # save plots of the track-based crop trajectories and PCs overlaid
        # on the flow field and trajectories from the grid-based crops
        make_all_plots(
            out_dir,
            dataset_name,
            diffae_grid_crops,
            traj_grids,
            flow_field_dict_grids,
            df_all_positions,
            traj_tracks,
        )


if __name__ == "__main__":
    ipython_cli_flexecute(main)
