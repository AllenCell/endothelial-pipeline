from pathlib import Path

import numpy as np

from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_model_manifest,
    load_model_config,
)
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.diffae_features.track_integration import (
    get_diffae_feats_liveseg_feats_merged_table,
    get_traj_and_flowfield,
)
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import make_all_plots


def main() -> None:
    out_dir = get_output_path(Path(__file__).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name_list = get_datasets_in_collection("pca_reference")

    for dataset_name in dataset_name_list:
        # create subdirectory to save track-based trajectories to
        out_subdir_traj = out_dir / "trajectories_track_based"
        out_subdir_traj.mkdir(parents=True, exist_ok=True)

        df_all_positions = get_diffae_feats_liveseg_feats_merged_table(dataset_name)
        if df_all_positions is None:
            print(f"Dataset {dataset_name} is missing one or more data tables. Skipping...")
            continue

        print("cleaning up merged table...")
        df_all_positions = df_all_positions.query("valid_points >= 120")
        df_all_positions.dropna(axis="index", how="any", subset="is_unique", inplace=True)

        # fit the PCA (uses the reference datasets)
        pca = fit_pca()

        # read in the grid crop-based diffae features
        model_name = sequence_to_scalar(df_all_positions["model_name"])
        model_config = load_model_config(model_name)
        model_manifest = get_model_manifest(dataset_name, model_config)
        diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

        # add the PC columns to the track-based DiffAE table
        # (the grid-based DiffAE table already has them, but
        # but I believe that the columns are named "feat_0",
        # "feat_1", etc. when they should be named "pc1",
        # "pc2", etc.)
        df_all_positions = project_manifest_to_pcs(df_all_positions, pca)

        # use the full set of datasets to be analyzed for the bounds
        model_manifest_list = [
            get_model_manifest(dataset_name, model_config) for dataset_name in dataset_name_list
        ]
        bounds = ddff.set_3d_bounds_from_data(model_manifest_list, pca)

        print("getting trajectory and flow field for grid-based crops...")
        traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
            diffae_grid_crops, bounds, load_precomputed_trajectories=None
        )

        print("getting trajectory and flow field for tracks-based crops...")
        traj_tracks, _ = get_traj_and_flowfield(
            df_all_positions, bounds, load_precomputed_trajectories=None
        )
        # save the trajectory data from the track-based crops
        np.save(out_subdir_traj / f"{dataset_name}_traj_tracks.npy", traj_tracks)

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
