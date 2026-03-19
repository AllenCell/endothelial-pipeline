import gc
import logging
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import color_palette
from tqdm import tqdm

from endo_pipeline.configs import get_latent_dim_from_config
from endo_pipeline.io import get_config_dict_from_mlflow, get_output_path, load_dataframe
from endo_pipeline.library.analyze.data_driven_flow_field import solve_ddff_ode
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    add_crop_index,
    add_description_column,
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    get_datasets_in_collection,
    get_latent_feature_column_names,
    get_traj_and_diff,
    project_features_to_pcs,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
from endo_pipeline.library.analyze.optical_flow_calculator import one_direction_vector_field_example
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.visualize.integration.track_integration_viz import (
    get_valid_slice_indexes,
    grid_vs_track_vec_angle_hist2d,
    grid_vs_track_vec_dot_prod_hist2d,
    overlay_flow_fields_on_histograms,
    plot_and_save_track_flow_field_deviations,
    plot_and_save_track_flow_field_dot_product_histogram,
    plot_grid_vs_tracks_flow_field,
    plot_pc_integrated_track_as_arrows,
)
from endo_pipeline.manifests import (
    ModelManifest,
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    get_model_location_for_run,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    MAX_PCS_TO_COMPUTE,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.flow_field_3d import (
    BIN_WIDTH_DEFAULTS,
    INIT_POINT_3D,
    KERNEL_BANDWIDTH,
    KERNEL_FUNCTION_NAME,
    TIME_STEP_IN_MINUTES,
    TRAJECTORY_TIME_SPAN,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def process_dataset_for_track_integration(
    dataset_name: str,
    collection_name_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    make_integrated_plots: bool = True,
) -> None:
    logger.info("Processing dataset: [ %s ]", dataset_name)

    out_subdir = get_output_path(__file__, dataset_name)

    # load and preprocess the different diffae manifests and PCA pipeline
    merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
        dataset_name=dataset_name,
        model_manifest_name=model_manifest_name,
        run_name=run_name,
        seg_feature_manifest_name=seg_feature_manifest_name,
        collection_name_for_pca=collection_name_for_pca,
    )

    # keep only the columns that are needed for the analysis to reduce memory usage
    cols_to_keep = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.SegData.LABEL,
        Column.CROP_INDEX,
        Column.DiffAEData.MODEL_MANIFEST,
        Column.SegData.TIME_HRS,
        Column.SegData.TIME_MINS,
        Column.TRACK_LENGTH,
    ] + [col for col in merged_feats_df.columns if "feat" in col or "pc" in col]
    merged_feats_df = merged_feats_df[cols_to_keep]

    # load or compute the trajectories and flow fields for the grid-based
    # and cell-centric crops
    traj_grids, flow_field_dict_grids, traj_tracks, flow_field_dict_tracks = (
        get_gridcrop_and_cellcentric_trajectories_and_flow_fields(
            dataset_name=dataset_name,
            merged_feats_df=merged_feats_df,
            diffae_grid_crops=diffae_grid_crops,
            bounds=bounds,
            trajectory_dir=out_subdir,
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
    grid_vs_track_vec_angle_hist2d(
        angles,
        out_subdir,
        filename=f"{dataset_name}_vecvec_angles",
        extent=(*ax.get_xlim(), *ax.get_ylim()),
    )

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
    grid_vs_track_vec_dot_prod_hist2d(
        dot_prod,
        out_subdir,
        filename=f"{dataset_name}_vecvec_dot_products",
        extent=(*ax.get_xlim(), *ax.get_ylim()),
    )

    # Compare the angles between grid crop PC vectors
    # and the PC vectors of a single track
    merged_feats_df["dpc1"] = merged_feats_df.groupby(Column.CROP_INDEX)["pc_1"].diff()
    merged_feats_df["dpc2"] = merged_feats_df.groupby(Column.CROP_INDEX)["pc_2"].diff()
    merged_feats_df["dt"] = merged_feats_df.groupby(Column.CROP_INDEX)[
        Column.SegData.TIME_MINS
    ].diff()

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
        columns=[["pc_1", "pc_2"]], data=get_approx_grid_bin(df.to_numpy()), index=df.index
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
        columns=[["pc_1", "pc_2"]], data=get_approx_grid_vec(df.to_numpy()), index=df.index
    )

    # Apply the partial functions to the DataFrame to get the approximate grid bin
    # and vector associated with each cell-centric PC1 and PC2 value
    merged_feats_df[["approx_bin_pc1", "approx_bin_pc2"]] = (
        merged_feats_df.groupby(Column.DATASET, as_index=False)
        .apply(lambda df: get_approx_grid_bin_from_df(df[["pc_1", "pc_2"]]))
        .droplevel(level=0)
    )
    merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
        merged_feats_df.groupby(Column.CROP_INDEX, as_index=False)
        .apply(lambda df: get_approx_grid_vec_from_df(df[["pc_1", "pc_2"]]))
        .droplevel(level=0)
    )

    # Compute the angle between the approximate grid vector
    # and the the vector from the cell-centric PC1 and PC2
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

    # get the dot products
    merged_feats_df["dot_product_grid_vs_cell"] = np.einsum(
        "ij,ij->i",
        merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]],
        merged_feats_df[["dpc1", "dpc2"]],
    )
    # also aggregate the dot products by crop index (i.e. unique track id across all positions)
    merged_feats_dot_prod_agg = (
        merged_feats_df.groupby("crop_index")["dot_product_grid_vs_cell"]
        .agg(["mean", "median"])
        .reset_index()
    )

    plot_title = "Mean per track"
    col_name = "mean"
    plot_and_save_track_flow_field_dot_product_histogram(
        features_dataframe=merged_feats_dot_prod_agg,
        feature_column_name=col_name,
        out_dir=out_subdir,
        filename=f"{dataset_name}_dot_product_grid_vs_cell_{col_name}",
        plot_title=plot_title,
    )

    plot_title = "Non-aggregated dot products"
    col_name = "dot_product_grid_vs_cell"
    plot_and_save_track_flow_field_dot_product_histogram(
        features_dataframe=merged_feats_df,
        feature_column_name=col_name,
        out_dir=out_subdir,
        filename=f"{dataset_name}_dot_product_grid_vs_cell_{col_name}",
        plot_title=plot_title,
    )

    if make_integrated_plots:
        # NOTE: this is a very memory-intensive operation despite my attempts to
        # reduce memory needs here, so if you change the minimum track duration
        # then expect the workflow to require a lot more memory or crash if you
        # don't have enough
        merged_feats_df = merged_feats_df.query("track_duration > 180")
        groups = merged_feats_df.groupby([Column.DATASET, Column.POSITION, Column.CROP_INDEX])

        i = 0
        for nm, df in tqdm(groups, desc=dataset_name):
            ds_nm, pos, tid = nm
            assert (
                tid % 1
            ) == 0, f"Track ID should be an integer or convertible to an integer. Got {tid}."
            hue_min = -1 * np.nanmax(merged_feats_df["dot_product_grid_vs_cell"].abs())
            hue_max = 1 * np.nanmax(merged_feats_df["dot_product_grid_vs_cell"].abs())
            hue_center = 0.0
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
                hue_min=hue_min,
                hue_max=hue_max,
                hue_center=hue_center,
                cmap_name="managua",
                hued_feat_name="dot_product_grid_vs_cell",
                track_alpha=0.5,
            )
            i += 1
            if i % 100 == 0:
                # force garbage collection to keep memory free when
                # creating plots from a loop every 100th iteration
                gc.collect()

    # overlay flow fields on the histograms of the data to see where
    # most of the data being used to extrapolate flow fields is
    overlay_flow_fields_on_histograms(
        dataset_name,
        out_subdir,
        diffae_grid_crops,
        merged_feats_df,
        v1_grids,
        v2_grids,
        g1_grids,
        g2_grids,
        v1_tracks,
        v2_tracks,
        g1_tracks,
        g2_tracks,
        slice_indexes,
    )


def add_normalized_time(
    df_all_positions: pd.DataFrame,
    time_col: str = Column.SegData.TIME_HRS,
) -> pd.DataFrame:
    """
    Add a column to the dataframe with normalized time values
    between 0 and 1 for each track_id in each position.

    Parameters
    ----------
    df_all_positions
        DataFrame containing all positions and tracks.
    time_col
        The name of the column containing time values.

    Returns
    -------
    :
        DataFrame with an additional column
        "normalized_time" containing the normalized time values between 0 and 1.
    """

    for _, df_pos in df_all_positions.groupby(Column.POSITION):
        for _, df_track in df_pos.groupby(Column.TRACK_ID):

            time_values = df_track[time_col].values.astype(np.float64)
            sorted_inds = np.argsort(time_values)
            time_values = time_values[sorted_inds]
            df_track = df_track.iloc[sorted_inds]

            start_time = np.min(time_values)
            end_time = np.max(time_values)

            normalized_time_values = np.divide(
                time_values - start_time,
                end_time - start_time,
                out=np.zeros_like(time_values, dtype=np.float64),
                where=(end_time - start_time) != 0,
            )

            normalized_time_values = np.clip(normalized_time_values, 0, 1)

            df_all_positions.loc[
                df_track.index,
                Column.SegData.NORMALIZED_TIME_PER_TRACK,
            ] = normalized_time_values

    return df_all_positions


def merge_diffae_feats_liveseg_feats_tables(
    diffae_tracking_df: pd.DataFrame,
    live_seg_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges the DiffAE tracking data with the live segmentation features data.

    Parameters
    ----------
        diffae_tracking_df (pd.DataFrame): DataFrame containing DiffAE tracking data.
        live_seg_feats_df (pd.DataFrame): DataFrame containing live segmentation features data.

    Returns
    -------
        pd.DataFrame: Merged DataFrame with DiffAE and live segmentation features.
    """
    dataset_name = sequence_to_scalar(diffae_tracking_df[Column.DATASET])
    logging.debug("processing the diffae tracking data...")
    # process the diffae tracking data
    track_is_unique = diffae_tracking_df.groupby(
        [Column.DATASET, Column.POSITION, Column.TIMEPOINT, Column.TRACK_ID]
    )[Column.TIMEPOINT].transform(lambda t: t.nunique() == t.size)
    if not track_is_unique.all():
        raise ValueError(
            "Found non-unique track_id and timepoint combinations in the diffae tracking data. "
            "Tracking data needs to be curated so that each position has unique Track IDs."
        )

    # add crop_index column (track_ids are not unique across positions but crop_index is)
    diffae_tracking_df = add_crop_index(df=diffae_tracking_df, crop_pattern="tracked")
    # add description column (e.g., 48hr_High)
    diffae_tracking_df = add_description_column(diffae_tracking_df, dataset_name, simple=True)

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merging_cols = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.ZARR_PATH,
    ]

    merged_feats_df = pd.merge(
        left=live_seg_feats_df,
        right=diffae_tracking_df,
        how="left",
        on=merging_cols,
        validate="one_to_one",
        suffixes=("_cdh5_seg", "_diffae_model"),
    )

    return merged_feats_df


def get_diffae_feats_liveseg_feats_merged_table(
    dataset_name: str,
    model_manifest: ModelManifest,
    run_name: str | None = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    filtered: bool = False,
) -> pd.DataFrame:
    """
    Get a merged dataframe with cell-centric DiffAE features and classical
    segmentation features.

    Parameters
    ----------
    dataset_name
        The name of the dataset to use.
    model_manifest
        The model manifest to use for DiffAE features.
    run_name
        The name of the run to use from the model manifest.
        If None, uses the most recent run.
    seg_feature_manifest_name
        The name of the segmentation feature manifest to use for classical features.

    Returns
    -------
    :
        The merged dataframe with DiffAE and segmentation features.
    """

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    tracked_dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="tracked"
    )
    diffae_track_manifest = load_dataframe_manifest(tracked_dataframe_manifest_name)
    diffae_track_location = get_dataframe_location_for_dataset(diffae_track_manifest, dataset_name)
    diffae_tracking_df = load_dataframe(diffae_track_location, delay=False)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_manifest = load_dataframe_manifest(seg_feature_manifest_name)
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df = load_dataframe(live_seg_location, delay=False)

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    if filtered:
        # filter the merged table
        merged_feats_df = merged_feats_df[merged_feats_df[Column.SegDataFilters.IS_INCLUDED]]

        # remove any rows that were not evaluated by the model and thus have no model_manifest_name
        merged_feats_df.dropna(
            axis="index", how="any", subset=Column.DiffAEData.MODEL_MANIFEST, inplace=True
        )

    return merged_feats_df


def get_traj_and_flowfield(
    df: pd.DataFrame,
    bounds: list,
    load_precomputed_trajectories: Path | None,
) -> tuple[np.ndarray, dict]:

    # set kernel params
    kernel_name = KERNEL_FUNCTION_NAME
    kernel_bw = KERNEL_BANDWIDTH

    # set time between frames in minutes
    dt = TIME_STEP_IN_MINUTES

    # time span for the ODE solver
    # units for time steps are in minutes
    # 48 hours in minutes =
    # 48 * 60 = 2880 time steps
    time_span = TRAJECTORY_TIME_SPAN

    # initial condition for the ODE solver
    # this is fixed across datasets /
    # shear stress conditions
    init = np.array(INIT_POINT_3D)

    bins, centers = get_bins(BIN_WIDTH_DEFAULTS, bin_limits=bounds)

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]

    # get list of per-crop trajectories and the corresponding
    # single-timepoint displacement vectors
    traj_list, d_traj_list = get_traj_and_diff(df, cols)

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel=KramersMoyalKernel(kernel_name, kernel_bw)
    )

    # get the vector field components from
    # the Kramers-Moyal coefficients
    grid = np.meshgrid(*centers, indexing="ij")
    drift_vector_field = [drift_km[..., i] for i in range(NUM_PCS_TO_ANALYZE)]
    flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

    if load_precomputed_trajectories is not None:
        logger.debug("Loading precomputed trajectories...")
        traj = np.load(load_precomputed_trajectories)
    else:
        # solve IVP, get back trajectory
        logger.debug("Trying to solve ODE...")
        traj = solve_ddff_ode(flow_field_dict, init, time_span)
        logger.debug("ODE solved.")

    return traj, flow_field_dict


def get_gridcrop_and_cellcentric_trajectories_and_flow_fields(
    dataset_name: str,
    merged_feats_df: pd.DataFrame,
    diffae_grid_crops: pd.DataFrame,
    bounds: list[float],
    trajectory_dir: Path,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    Get the trajectories and flow fields for the grid-based and cell-centric crops.
    This function is called after loading and preprocessing the manifests.
    The function looks for precomputed trajectories in trajectory_dir and loads them
    from there if found. If not found then they will be computed and saved to that location.
    The names of the files that it looks for are:
    - {dataset_name}_traj_grids.npy for grid-based crops
    - {dataset_name}_traj_tracks.npy for cell-centric crops
    """
    logger.info("Getting trajectories and flow fields for grid-based and cell-centric crops...")

    # try to load the grid crop-based  data for the cell-centric
    #  crops or, if needed, compute and save them
    precomputed_trajectories_path = trajectory_dir / f"{dataset_name}_traj_grids.npy"
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

    # try to load the trajectory data for the cell-centric crops or,
    # if needed, compute and save them
    precomputed_trajectories_path = trajectory_dir / f"{dataset_name}_traj_tracks.npy"
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


def get_vector_vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return angle_rad


def get_vector_vector_angle_fast(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot_prod = np.einsum("ij,ij->i", v1, v2)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    angle_rad = np.arccos(dot_prod / (norm1 * norm2))
    return angle_rad


def get_approx_vec_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(
        list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes], strict=True))
    )
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    v1_grids_approx = v1_grids[slice_indexes][min_idx]
    v2_grids_approx = v2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(v1_grids_approx.tolist(), v2_grids_approx.tolist(), strict=True)))


def get_approx_point_from_grid(
    pc1_pc2_points: np.ndarray,
    g1_grids: np.ndarray,
    g2_grids: np.ndarray,
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:

    # create a distance mapping
    point_grids_pc1pc2 = np.asarray(
        list(zip(g1_grids[slice_indexes], g2_grids[slice_indexes], strict=True))
    )
    pc1_pc2_points = np.expand_dims(pc1_pc2_points, axis=0)
    point_grids_pc1pc2 = np.expand_dims(point_grids_pc1pc2, axis=1)
    dists = np.linalg.norm(point_grids_pc1pc2 - pc1_pc2_points, axis=-1)

    # get the index of the closest point
    min_idx = np.argmin(dists, axis=0)
    g1_grids_approx = g1_grids[slice_indexes][min_idx]
    g2_grids_approx = g2_grids[slice_indexes][min_idx]

    return np.array(tuple(zip(g1_grids_approx.tolist(), g2_grids_approx.tolist(), strict=True)))


def get_vector_angles_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the angles of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(
        list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids), strict=True))
    )
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks), strict=True))
    )
    ang_full = get_vector_vector_angle_fast(vecs_grids, vecs_tracks)
    ang_arr = ang_full.reshape(v1_grids.shape)
    angles = ang_arr[slice_indexes].reshape(my_shape)
    return angles


def get_vector_dot_products_as_grid(
    v1_grids: np.ndarray,
    v2_grids: np.ndarray,
    v3_grids: np.ndarray,
    v1_tracks: np.ndarray,
    v2_tracks: np.ndarray,
    v3_tracks: np.ndarray,
    slice_indexes: tuple[np.ndarray[Any, np.dtype[np.signedinteger[Any]]], ...],
) -> np.ndarray:
    """Get the dot products of the vectors as a grid."""
    my_shape = [len(np.unique(slice_indexes[i])) for i in range(len(slice_indexes))]

    vecs_grids = np.asarray(
        list(zip(np.ravel(v1_grids), np.ravel(v2_grids), np.ravel(v3_grids), strict=True))
    )
    vecs_tracks = np.asarray(
        list(zip(np.ravel(v1_tracks), np.ravel(v2_tracks), np.ravel(v3_tracks), strict=True))
    )
    dot_prod_full = np.einsum("ij,ij->i", vecs_grids, vecs_tracks)
    dot_prod_arr = dot_prod_full.reshape(v1_grids.shape)
    dot_prod = dot_prod_arr[slice_indexes].reshape(my_shape)
    return dot_prod


def make_angular_deviation_test(out_dir: Path) -> None:
    test_flow_field = one_direction_vector_field_example()

    test_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ]
    )

    test_points = np.array(
        [
            [-8.0, -4.0],
            [-6.0, -3.0],
            [-4.0, -2.0],
            [-2.0, -1.0],
            [2.0, 1.0],
            [4.0, 2.0],
            [6.0, 3.0],
            [8.0, 4.0],
        ]
    )

    slice_indexes = np.where(np.ones_like(test_flow_field[0][1]))
    test_flow_field_points = get_approx_point_from_grid(
        test_points,
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        slice_indexes,
    )

    test_flow_field_vectors = get_approx_vec_from_grid(
        test_vectors,
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        slice_indexes,
    )

    test_angular_deviation = get_vector_vector_angle_fast(test_flow_field_vectors, test_vectors)
    test_angular_deviation_deg = np.rad2deg(test_angular_deviation)

    cmap = color_palette("dark:red", as_cmap=True)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.quiver(
        test_flow_field[1][0],
        test_flow_field[1][1],
        test_flow_field[0][0],
        test_flow_field[0][1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color="lightgrey",
    )
    ax.quiver(
        test_flow_field_points[:, 0],
        test_flow_field_points[:, 1],
        test_flow_field_vectors[:, 0],
        test_flow_field_vectors[:, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color="grey",
    )
    ax.quiver(
        test_points[:, 0],
        test_points[:, 1],
        test_vectors[:, 0],
        test_vectors[:, 1],
        scale_units="xy",
        angles="xy",
        scale=1,
        units="width",
        width=0.005,
        alpha=1,
        color=cmap(np.abs(test_angular_deviation_deg) / 180.0),  # convert angle to color
    )
    ax.set_xlim(-9, 9)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("Angular deviation from\nflow field test")
    fig.savefig(
        out_dir / "get_angular_deviation_deg_test.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return


def get_preprocessed_manifests_and_km_bounds(
    dataset_name: str,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    collection_name_for_pca: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    num_pcs: int = MAX_PCS_TO_COMPUTE,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load and process the DiffAE and live segmentation feature manifests for a given dataset.
    If no `datasets_for_bounds` are provided, it uses the reference datasets plus dataset_name
    to compute the bounds for the PCA projection. In my experience using only the dataset_name
    for the bounds has sometimes caused the solver to hang, perhaps due to overly restrictive
    bounds.

    Parameters
    ----------
    dataset_name
        The name of the dataset to load and process.
    model_manifest
        The model manifest to use for loading the DiffAE features.
    run_name
        The run name to use for loading the DiffAE features. If None, the most recent
        run will be used.
    seg_feature_manifest_name
        The name of the manifest containing segmentation features.
    collection_name_for_pca
        The name of the dataset collection to use for fitting the PCA. Defaults to
        DEFAULT_PCA_DATASET_COLLECTION_NAME.
    num_pcs
        The number of principal components to use for the PCA projection. If None, the minimum of
        NUM_PCS_TO_ANALYZE and the number of latent dimensions will be used.

    Returns
    -------
    :
        A tuple containing the merged DiffAE and live segmentation features DataFrame,
        the grid crop-based DiffAE features DataFrame, and the PCA bounds.
    """
    logger.info(f"Loading and processing manifests for dataset: {dataset_name}")

    # get the cell-centric merged DiffAE + segmentation feature table
    model_manifest = load_model_manifest(model_manifest_name)

    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(
        dataset_name=dataset_name,
        model_manifest=model_manifest,
        run_name=run_name,
        seg_feature_manifest_name=seg_feature_manifest_name,
        filtered=False,  # do not filter timepoints (need all timepoints for the TFE workflow)
    )

    # check the model information matches the default values and what will be used for the PCA
    model_manifest_name_used_for_latent_feats = sequence_to_scalar(
        merged_feats_df[Column.DiffAEData.MODEL_MANIFEST].dropna()
    )
    model_run_name_used_for_latent_feats = sequence_to_scalar(
        merged_feats_df[Column.DiffAEData.RUN_NAME].dropna()
    )

    if (DEFAULT_MODEL_MANIFEST_NAME != model_manifest_name_used_for_latent_feats) or (
        DEFAULT_MODEL_RUN_NAME != model_run_name_used_for_latent_feats
    ):
        raise ValueError(
            """"The model manifest name or run name used to produce the DiffAE
            features found in the merged features dataframe does not match the
            expected default values being used for the PCA.
            """
        )

    # load the grid crop-based diffae features manifest
    grid_diffae_feat_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    # fit the PCA
    model_location = get_model_location_for_run(model_manifest, run_name)
    model_config = get_config_dict_from_mlflow(model_location.mlflowid)
    num_latent_dims = get_latent_dim_from_config(model_config)
    diffae_feature_column_names = get_latent_feature_column_names(num_latent_dims)

    pca = fit_pca(
        dataset_collection_name=collection_name_for_pca,
        dataframe_manifest_name=grid_diffae_feat_manifest_name,
        num_pcs=num_pcs,
    )

    # The PCA cannot take in NaN values, so subset the dataframe by the
    # model_manifest_name column (which has the model_manifest_name if the
    # eval_diffae_tracked.py workflow was evaluated on that row, otherwise
    # it has NaN as an entry) and then get the PCs for the subset of data
    # with DiffAE features only, once that is done we merge the original
    # dataframe and the DiffAE features dataframe.
    # This way we avoid passing NaN values to the PCA but still return the
    # full dataframe with all timepoints which is required by the TFE workflow.
    merged_feats_df_subset = merged_feats_df[
        [Column.DiffAEData.MODEL_MANIFEST, *diffae_feature_column_names]
    ].dropna(axis="index", how="any", subset=Column.DiffAEData.MODEL_MANIFEST)
    tracked_diffae_feats_df = project_features_to_pcs(
        merged_feats_df_subset, pca, diffae_feature_column_names
    )
    tracked_diffae_feats_df = tracked_diffae_feats_df.drop(
        columns=[Column.DiffAEData.MODEL_MANIFEST, *diffae_feature_column_names]
    )
    tracked_diffae_feats_df = tracked_diffae_feats_df.assign(
        collection_name_for_pca=collection_name_for_pca
    )
    tracked_diffae_feats_df = tracked_diffae_feats_df.assign(
        datasets_used_for_pca=[get_datasets_in_collection(collection_name_for_pca)]
        * len(tracked_diffae_feats_df)
    )

    # tracked_diffae_feats_df retains the indexing of merged_feats_df, so we
    # can merge on the index safely
    merged_feats_df = pd.merge(
        left=merged_feats_df,
        right=tracked_diffae_feats_df,
        how="left",
        left_index=True,
        right_index=True,
        validate="one_to_one",
    ).reset_index(drop=True)

    # read in the grid crop-based diffae features
    grid_diffae_manifest = load_dataframe_manifest(grid_diffae_feat_manifest_name)
    diffae_grid_crops = get_dataframe_for_dynamics_workflows(
        dataset_name=dataset_name,
        manifest=grid_diffae_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
    )

    # get bounds for plotting and flow field estimation
    # based on this dataset only
    bounds = get_bounds_from_data([dataset_name], grid_diffae_manifest, pca)

    # lastly, add a normalized version of the "time_hours" column
    merged_feats_df = add_normalized_time(merged_feats_df)

    return merged_feats_df, diffae_grid_crops, bounds


def get_and_save_pc_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> None:
    """Loads the cell-centric DiffAE + segmentation features merged table, computes the PCs, and
    then saves the updated merged table with the PCs as a parquet file.
    """

    out_dir = get_output_path(__file__)

    merged_feats_df = get_preprocessed_manifests_and_km_bounds(dataset_name=dataset_name)[0]

    filename = f"{dataset_name}_pc_diffae_seg_feats_merged.parquet"

    merged_feats_df.to_parquet(out_dir / filename)


def load_preprocessed_dataframes_and_km_bounds(
    dataset_name: str,
    cell_centric_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    num_pcs: int = MAX_PCS_TO_COMPUTE,
    delay: bool = True,
) -> tuple[pd.DataFrame | dd.DataFrame, pd.DataFrame | dd.DataFrame, list]:
    """
    Load the preprocessed pc-diffae-seg-merged parquet file for a given dataset.

    Parameters
    ----------
    dataset_name
        The name of the dataset to load.
    cell_centric_manifest_name
        The name of the manifest containing the cell-centric pc-diffae-seg-merged features.
    num_pcs
        The number of principal components to use for the PCA projection. This only
        applies to the grid crop-based diffae features dataframe (the cell-centric
        features dataframe has 100 PCs computed already).
    delay
        Whether to delay the loading of the dataframe (Dask DataFrame) or not (Pandas DataFrame).

    Returns
    -------
    :
        The loaded dataframe with pc-diffae-seg-merged data.
    """
    # load the pc-diffae-seg-merged parquet file
    cell_centric_feats_manifest = load_dataframe_manifest(cell_centric_manifest_name)
    cell_centric_feats_location = get_dataframe_location_for_dataset(
        cell_centric_feats_manifest, dataset_name
    )
    cell_centric_feats_df = load_dataframe(cell_centric_feats_location, delay=True)
    cell_centric_feats_df = cell_centric_feats_df.reset_index(drop=True)

    # get the grid crop-based diffae features
    # get the model information
    model_manifest_name = sequence_to_scalar(
        cell_centric_feats_df["model_manifest_name"].compute().dropna()
    )
    run_name = sequence_to_scalar(cell_centric_feats_df["run_name"].compute().dropna())
    model_manifest = load_model_manifest(model_manifest_name)

    # get the datasets used to calculate the PCA in the cell-centric features
    collection_name_for_pca = sequence_to_scalar(
        cell_centric_feats_df["collection_name_for_pca"].compute().dropna()
    )

    # load the grid crop-based diffae features manifest
    grid_diffae_feat_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    # get the fitted PCA
    pca = fit_pca(
        dataset_collection_name=collection_name_for_pca,
        dataframe_manifest_name=grid_diffae_feat_manifest_name,
        num_pcs=num_pcs,
    )
    # read in the grid crop-based diffae features
    grid_diffae_manifest = load_dataframe_manifest(grid_diffae_feat_manifest_name)
    diffae_grid_crops = get_dataframe_for_dynamics_workflows(
        dataset_name,
        grid_diffae_manifest,
        pca=pca,
        include_cell_piling=False,
        include_not_steady_state=False,
    )

    # get bounds for plotting and flow field estimation
    bounds = get_bounds_from_data([dataset_name], grid_diffae_manifest, pca)

    if delay is False:
        cell_centric_feats_df = cell_centric_feats_df.compute()  # type: ignore

    return cell_centric_feats_df, diffae_grid_crops, bounds
