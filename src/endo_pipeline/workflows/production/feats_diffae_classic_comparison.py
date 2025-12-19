"""This workflow outputs plots of track-based cell trajectories integrated with grid-based DiffAE flow fields."""

from endo_pipeline.cli import Datasets
from endo_pipeline.settings import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)


def main(
    # dataset_collection_name: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    datasets: Datasets | None = None,
    datasets_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    n_cores: int = 30,
) -> None:

    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.integration.track_integration import (
        get_gridcrop_and_cellcentric_trajectories_and_flow_fields,
        get_preprocessed_manifests_and_km_bounds,
    )
    from endo_pipeline.library.visualize.integration.track_integration_viz import make_all_plots
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings import NUM_PCS_TO_ANALYZE

    out_dir = get_output_path(__file__)
    if datasets is None:
        datasets = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

    model_manifest = load_model_manifest(model_manifest_name)

    # create subdirectory to save track-based trajectories to
    out_subdir_traj = out_dir / "trajectories_track_based"
    out_subdir_traj.mkdir(parents=True, exist_ok=True)

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
        make_all_plots(
            out_dir,
            dataset_name,
            diffae_grid_crops,
            traj_grids,
            flow_field_dict_grids,
            df_all_positions,
            traj_tracks,
            n_cores=n_cores,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
