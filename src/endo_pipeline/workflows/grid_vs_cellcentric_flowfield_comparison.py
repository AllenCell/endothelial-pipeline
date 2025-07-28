import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.endo_pipeline.configs import (
    get_model_manifest,
    load_dataset_collection_config,
    load_model_config,
)
from src.endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from src.endo_pipeline.io import configure_logging, get_output_path
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_features.track_integration import (
    get_approx_point_from_grid,
    get_approx_vec_from_grid,
    get_diffae_feats_liveseg_feats_merged_table,
    get_traj_and_flowfield,
    get_vector_angles_as_grid,
    get_vector_dot_products_as_grid,
    get_vector_vector_angle_fast,
    make_angular_deviation_test,
)
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import plot_one_slice_quiver
from src.endo_pipeline.library.visualize.diffae_features.track_integration_viz import (
    get_valid_slice_indexes,
    grid_vs_track_vec_angle_hist2d,
    grid_vs_track_vec_dot_prod_hist2d,
    plot_and_save_track_flow_field_deviations,
    plot_grid_vs_tracks_flow_field,
    plot_pc_integrated_track_as_arrows,
)

logger = logging.getLogger(__name__)


def get_preprocessed_manifests_and_km_bounds(
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Load and process the DiffAE and live segmentation feature manifests for a given dataset.
    """
    logger.info(f"Loading and processing manifests for dataset: {dataset_name}")

    # load the tables
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(dataset_name, filtered=True)

    # keep only the columns that are needed for the analysis to reduce memory usage
    cols_to_keep = [
        "dataset_name",
        "position",
        "position_as_str",
        "track_id",
        "label",
        "crop_index",
        "mlflow_id",
        "model_name",
        "image_index",
        "frame_number",
        "time_hours",
        "time_minutes",
        "track_duration",
    ] + [col for col in merged_feats_df.columns if "feat" in col]

    merged_feats_df = merged_feats_df[cols_to_keep]

    # fit the PCA (uses the reference datasets)
    pca = fit_pca()

    # read in the grid crop-based diffae features
    model_name = merged_feats_df["model_name"].unique()[0]
    model_config = load_model_config(model_name)
    model_manifest = get_model_manifest(dataset_name, model_config)
    diffae_grid_crops = get_manifest_for_dynamics_workflows(model_manifest, pca)

    # add the PC columns to the track-based DiffAE table
    # (the grid-based DiffAE table already has them, but
    # but I believe that the columns are named "feat_0",
    # "feat_1", etc. when they should be named "pc1",
    # "pc2", etc.)
    merged_feats_df = project_manifest_to_pcs(merged_feats_df, pca)

    # use the full set of datasets to be analyzed for the bounds
    datasets_for_bounds = [
        "20241120_20X",
        "20250409_20X",
        "20241217_20X",
        "20250428_20X",
        "20250319_20X",
        "20250326_20X",
    ]

    model_manifest_list = [
        get_model_manifest(dataset_name, model_config) for dataset_name in datasets_for_bounds
    ]
    bounds = ddff.set_3d_bounds_from_data(model_manifest_list, pca)

    return merged_feats_df, diffae_grid_crops, bounds


def get_trajectories_and_flow_fields(
    dataset_name: str,
    merged_feats_df: pd.DataFrame,
    diffae_grid_crops: pd.DataFrame,
    bounds: tuple[float, float, float, float, float, float],
    out_subdir: Path,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    Get the trajectories and flow fields for the grid-based and cell-centric crops.
    This function is called after loading and preprocessing the manifests.
    """
    logger.info("Getting trajectories and flow fields for grid-based and cell-centric crops...")
    # This function will be defined in the main processing function
    # to handle the specific dataset being processed.
    precomputed_trajectories_path = out_subdir / f"{dataset_name}_traj_grids.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for grid-based crops...")
    # This takes about 2 minutes to compute if not loading precomputed
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        df=diffae_grid_crops,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the grid-based crops...")
        np.save(precomputed_trajectories_path, traj_grids)

    precomputed_trajectories_path = out_subdir / f"{dataset_name}_traj_tracks.npy"
    if not precomputed_trajectories_path.exists():
        logger.debug("Precomputed trajectories not found, will compute them...")
        load_precomputed_trajectories = None
    else:
        load_precomputed_trajectories = precomputed_trajectories_path

    logger.debug("getting trajectory and flow field for tracks-based crops...")
    # This takes about 5 minutes to compute if not loading precomputed
    traj_tracks, flow_field_dict_tracks = get_traj_and_flowfield(
        df=merged_feats_df,
        bounds=bounds,
        load_precomputed_trajectories=load_precomputed_trajectories,
    )

    if load_precomputed_trajectories is None:
        logger.debug("saving the trajectory data from the track-based crops...")
        np.save(precomputed_trajectories_path, traj_tracks)

    return traj_grids, flow_field_dict_grids, traj_tracks, flow_field_dict_tracks


def process_dataset(dataset_name: str, make_integrated_plots: bool = True) -> None:
    logger.info(f"Processing dataset: {dataset_name}")

    out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)
    configure_logging(out_subdir, logger, verbose=True)

    # load and preprocess the different diffae manifests and PCA pipeline
    merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
        dataset_name
    )

    # load or compute the trajectories and flow fields for the grid-based
    # and cell-centric crops
    traj_grids, flow_field_dict_grids, traj_tracks, flow_field_dict_tracks = (
        get_trajectories_and_flow_fields(
            dataset_name=dataset_name,
            merged_feats_df=merged_feats_df,
            diffae_grid_crops=diffae_grid_crops,
            bounds=bounds,
            out_subdir=out_subdir,
        )
    )

    # get the slice indexes to use for plotting the flow fields
    # (we will be setting PC3 to a constant, i.e. the z-axis here)
    _, slice_indexes = get_valid_slice_indexes(diffae_grid_crops, traj_grids, flow_field_dict_grids)

    # get flow field vectors and grid points to plot
    v1_grids, v2_grids, v3_grids = flow_field_dict_grids["vectors"]
    g1_grids, g2_grids, g3_grids = flow_field_dict_grids["grid"]
    v1_tracks, v2_tracks, v3_tracks = flow_field_dict_tracks["vectors"]
    g1_tracks, g2_tracks, g3_tracks = flow_field_dict_tracks["grid"]

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the full resolution:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_full_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,
        ds=1,
        scale=60,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the quiver slices for the grid-based and cell-centric crops
    # at the standard/default resolution and include the fixed points
    # for both the grid and cell-centric crops:
    out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_partial_quiver.png"
    fig, ax = plot_grid_vs_tracks_flow_field(
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes=slice_indexes,
    )
    # add the grid crop based fixed point from the trajectory:
    ax.scatter(
        traj_grids[-1, 0],
        traj_grids[-1, 1],
        s=250,
        color="cyan",
        marker="*",
        lw=1,
        edgecolor="darkblue",
        zorder=10,
    )
    # add the cell-centric crop based fixed point from the trajectory:
    ax.scatter(
        traj_tracks[-1, 0],
        traj_tracks[-1, 1],
        s=250,
        color="yellow",
        marker="*",
        lw=1,
        edgecolor="darkred",
        zorder=10,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the angular deviation between the grid and cell-centric crop-based
    # flow field vectors:
    angles = get_vector_angles_as_grid(
        v1_grids,
        v2_grids,
        v3_grids,
        v1_tracks,
        v2_tracks,
        v3_tracks,
        slice_indexes,
    )
    out_path = out_subdir / f"{dataset_name}_vecvec_angles.png"
    grid_vs_track_vec_angle_hist2d(angles, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Plot the dot product between the grid and cell-centric crop-based
    dot_prod = get_vector_dot_products_as_grid(
        v1_grids,
        v2_grids,
        v3_grids,
        v1_tracks,
        v2_tracks,
        v3_tracks,
        slice_indexes,
    )
    out_path = out_subdir / f"{dataset_name}_vecvec_dot_products.png"
    grid_vs_track_vec_dot_prod_hist2d(dot_prod, out_path, extent=(*ax.get_xlim(), *ax.get_ylim()))

    # Compare the angles between grid crop PC vectors
    # and the PC vectors of a single track
    merged_feats_df["dpc1"] = merged_feats_df.groupby("crop_index")["pc1"].diff()
    merged_feats_df["dpc2"] = merged_feats_df.groupby("crop_index")["pc2"].diff()
    merged_feats_df["dt"] = merged_feats_df.groupby("crop_index")["time_minutes"].diff()

    # create partial functions from get_approx_point_from_grid to pass
    # along to the groupby.apply() method
    get_approx_grid_bin = lambda pc1_pc2_arr: get_approx_point_from_grid(
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,
    )
    get_approx_grid_bin_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_bin(df.to_numpy()), index=df.index
    )

    get_approx_grid_vec = lambda pc1_pc2_arr: get_approx_vec_from_grid(
        pc1_pc2_arr,
        g1_grids,
        g2_grids,
        v1_grids,
        v2_grids,
        slice_indexes,
    )
    get_approx_grid_vec_from_df = lambda df: pd.DataFrame(
        columns=[["pc1", "pc2"]], data=get_approx_grid_vec(df.to_numpy()), index=df.index
    )

    # Apply the partial functions to the DataFrame to get the approximate grid bin
    # and vector associated with each cell-centric PC1 and PC2 value
    merged_feats_df[["approx_bin_pc1", "approx_bin_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_bin_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )
    merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_vec_from_df(df[["pc1", "pc2"]]))
        .droplevel(level=0)
    )

    # Compute the angle between the approximate grid vector
    #  and the the vector from the cell-centric PC1 and PC2
    # both in radians and degrees
    merged_feats_df["track_angle_deviation_rad"] = get_vector_vector_angle_fast(
        merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]].values,
        merged_feats_df[["dpc1", "dpc2"]].values,
    )
    merged_feats_df["track_angular_deviation_deg"] = merged_feats_df[
        "track_angle_deviation_rad"
    ].transform(np.rad2deg)

    merged_feats_df["pc1_pc2_vec_mag"] = np.linalg.norm(
        merged_feats_df[["dpc1", "dpc2"]].values, axis=1
    )

    # group dataframe by a combination of dataset, position, and crop index
    # note that we have replaced the track id with the crop index in this
    # case because the crop index is unique throughout all 6 positions,
    # whereas the track id is only unique within a single position
    mean_track_deviation_dfs = (
        merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"])[
            ["track_angular_deviation_deg", "pc1_pc2_vec_mag"]
        ]
        .agg("mean")
        .reset_index()
    )

    plot_and_save_track_flow_field_deviations(
        mean_track_deviation_dfs=mean_track_deviation_dfs,
        out_subdir=out_subdir,
        dataset_name=dataset_name,
    )

    if make_integrated_plots:
        merged_feats_df = merged_feats_df.query("track_duration > 120")
        groups = merged_feats_df.query("track_duration > 120").groupby(
            ["dataset_name", "position_as_str", "crop_index"]
        )

        i = 0
        for nm, df in tqdm(groups, desc=dataset_name):
            ds_nm, pos, tid = nm
            assert (
                tid % 1
            ) == 0, f"Track ID should be an integer or convertible to an integer. Got {tid}."
            plot_pc_integrated_track_as_arrows(
                dataset_name=str(ds_nm),
                position_name=str(pos),
                track_id=int(tid),
                df=df,
                v1_grids=v1_grids,
                v2_grids=v2_grids,
                g1_grids=g1_grids,
                g2_grids=g2_grids,
                slice_indexes=slice_indexes,
                out_subdir=out_subdir,
            )
            i += 1
            if i % 100 == 0:
                # force garbage collection to keep memory free when
                # creating plots from a loop every 100th iteration
                gc.collect()

    return


def main() -> None:
    dataset_name_list = load_dataset_collection_config("pca_reference").datasets

    for dataset_name in dataset_name_list:
        logger.info(f"Processing {dataset_name}...")
        process_dataset(dataset_name, make_integrated_plots=True)

    # create a test flow field and test set of vectors
    # to check that the angular deviation calculation
    # works as expected
    out_dir = get_output_path(Path(__file__).stem, include_timestamp=False)
    make_angular_deviation_test(out_dir)


if __name__ == "__main__":
    main()
