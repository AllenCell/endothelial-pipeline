import gc
import logging
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import color_palette
from tqdm import tqdm

from endo_pipeline.configs.dataset_config_io import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, save_plot_to_path
from endo_pipeline.library.analyze.data_driven_flow_field import solve_ddff_ode
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    check_required_columns_in_dataframe,
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    get_pc_column_names,
    get_traj_and_diff,
)
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins, get_bounds_from_data
from endo_pipeline.library.analyze.optical_flow_calculator import one_direction_vector_field_example
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.library.visualize.diffae_features.pplane import STABILITY_COLOR_DICT
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
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.manifests.model_manifest_io import load_model_manifest
from endo_pipeline.manifests.model_manifest_utils import get_feature_dataframe_manifest_name
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    MAX_PCS_TO_COMPUTE,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.dynamics_workflows import (
    DYNAMICS_COLUMN_NAMES,
    PERIOD_THETA_RESCALED,
    RESCALE_THETA,
)
from endo_pipeline.settings.flow_field_3d import (
    BIN_WIDTH_DEFAULTS,
    INIT_POINT_3D,
    KERNEL_BANDWIDTH,
    KERNEL_FUNCTION_NAME,
    TIME_STEP_IN_MINUTES,
    TRAJECTORY_TIME_SPAN,
)
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    STABILITY_COLUMN_NAME,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_COLUMNS_TO_DROP,
    DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_FILTERED,
    DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_UNFILTERED,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME,
    DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def process_dataset_for_track_integration(
    dataset_name: str,
    merged_cellcentric_features_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
    diffae_grid_manifest_name: str = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
    make_integrated_plots: bool = True,
) -> None:
    logger.info("Processing dataset: [ %s ]", dataset_name)

    out_subdir = get_output_path(__file__, dataset_name)

    # load the track-based diffae + segmentation feature merged manifest
    # and the grid-based diffae manifest
    merged_feats_manifest = load_dataframe_manifest(merged_cellcentric_features_manifest_name)
    merged_feats_location = get_dataframe_location_for_dataset(merged_feats_manifest, dataset_name)
    merged_feats_df = load_dataframe(merged_feats_location, delay=True)

    diffae_grid_manifest = load_dataframe_manifest(diffae_grid_manifest_name)
    diffae_grid_location = get_dataframe_location_for_dataset(diffae_grid_manifest, dataset_name)
    diffae_grid_df = load_dataframe(diffae_grid_location, delay=True)

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
            diffae_grid_crops=diffae_grid_df,
            trajectory_dir=out_subdir,
        )
    )

    # get the slice indexes to use for plotting the flow fields
    # (we will be setting PC3 to a constant, i.e. the z-axis here)
    _, slice_indexes = get_valid_slice_indexes(diffae_grid_df, traj_grids, flow_field_dict_grids)

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
        diffae_grid_df,
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

    logging.debug("merging segmentation properties and track-based DiffAE data...")
    merging_cols = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.ZARR_PATH,
    ]
    if Column.TRACK_LENGTH in diffae_tracking_df.columns:
        merging_cols.append(Column.TRACK_LENGTH)

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
    classic_segmentation_feature_manifest_name: str,
    diffae_tracked_feature_manifest_name: str,
    filter_columns: bool = False,
    additional_columns_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Get a merged dataframe with cell-centric DiffAE features and classical
    segmentation features.

    Parameters
    ----------
    dataset_name
        The name of the dataset to use.
    classic_segmentation_feature_manifest_name
        The name of the classic segmentation feature manifest to use.
    diffae_tracked_feature_manifest_name
        The name of the DiffAE tracked feature manifest to use.
    filter_columns
        Whether or not to pare down the columns in the returned merged dataframe.

    Returns
    -------
    :
        The merged dataframe with DiffAE and segmentation features.
    """

    # read in the segmentation-based diffae features if available
    logging.debug("loading diffae features from tracking data...")
    diffae_track_manifest = load_dataframe_manifest(diffae_tracked_feature_manifest_name)
    diffae_track_location = get_dataframe_location_for_dataset(diffae_track_manifest, dataset_name)
    diffae_tracking_df = load_dataframe(diffae_track_location, delay=False)

    # drop any pc columns after the 100th one
    all_pc_col_names = get_pc_column_names("all")
    first_100_pc_col_names = get_pc_column_names("first_100_pcs")
    pc_cols_to_drop = sorted(set(all_pc_col_names) - set(first_100_pc_col_names))

    diffae_tracking_df = diffae_tracking_df.drop(columns=pc_cols_to_drop)

    # load the tracking data of the measured features and merge them
    logging.debug("loading segmentation property data...")
    live_seg_manifest = load_dataframe_manifest(classic_segmentation_feature_manifest_name)
    live_seg_location = get_dataframe_location_for_dataset(live_seg_manifest, dataset_name)
    live_seg_feats_df = load_dataframe(live_seg_location, delay=False)

    # merge the two tables
    merged_feats_df = merge_diffae_feats_liveseg_feats_tables(diffae_tracking_df, live_seg_feats_df)

    if filter_columns:
        # filter the merged table
        merged_feats_df = merged_feats_df[merged_feats_df[Column.SegDataFilters.IS_INCLUDED]]

        # remove any rows that were not evaluated by the model and thus have no model_manifest_name
        merged_feats_df.dropna(
            axis="index", how="any", subset=Column.DiffAEData.MODEL_MANIFEST, inplace=True
        )

        # remove columns that were kept for workflow validations
        default_cols_to_drop = [
            col for col_grp in DEFAULT_COLUMNS_TO_DROP.values() for col in col_grp
        ]
        nuclei_intens_cols = [
            col
            for col in merged_feats_df.columns
            if Column.SegDataWorkflowVerification.NUCLEI_INTENSITY_COLUMN_PREFIX in col
        ]
        additional_cols_to_drop = additional_columns_to_drop or []

        cols_to_drop = [
            *default_cols_to_drop,
            *nuclei_intens_cols,
            *additional_cols_to_drop,
        ]

        merged_feats_df.drop(columns=cols_to_drop, inplace=True)

    return merged_feats_df.reset_index(drop=True)


def get_traj_and_flowfield(
    df: pd.DataFrame,
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

    # get the columns to use for calculating trajectories
    # and flow fields.
    cols = DIFFAE_PC_COLUMN_NAMES[:NUM_PCS_TO_ANALYZE]

    bins, centers = get_bins(BIN_WIDTH_DEFAULTS, data=df[cols].to_numpy())

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


def get_merged_pc_and_seg_feature_tables(
    dataset_name: str,
    classic_segmentation_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    diffae_tracked_feature_manifest_name_unfiltered: str = DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_UNFILTERED,
    diffae_tracked_feature_manifest_name_filtered: str = DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_FILTERED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge the track-based DiffAE and live segmentation feature tables for a given dataset.

    Parameters
    ----------
    dataset_name
        The name of the dataset to load and process.
    classic_segmentation_feature_manifest_name
        The manifest name to use for loading the classic segmentation features.
    diffae_tracked_feature_manifest_name_unfiltered
        The manifest name to load the unfiltered DiffAE-based features for cell-centric crops.
    diffae_tracked_feature_manifest_name_filtered
        The manifest name to load the filtered DiffAE-based features for cell-centric crops.

    Returns
    -------
    :
        A tuple containing both an unfiltered and filtered version of the the merged DiffAE
        and live segmentation features DataFrames.
    """
    logger.info(f"Loading and processing manifests for dataset: {dataset_name}")

    # get the cell-centric merged DiffAE + segmentation feature table for unfiltered data
    merged_feats_df = get_diffae_feats_liveseg_feats_merged_table(
        dataset_name=dataset_name,
        classic_segmentation_feature_manifest_name=classic_segmentation_feature_manifest_name,
        diffae_tracked_feature_manifest_name=diffae_tracked_feature_manifest_name_unfiltered,
    )

    # repeat for filtered data
    merged_feats_df_filtered = get_diffae_feats_liveseg_feats_merged_table(
        dataset_name=dataset_name,
        classic_segmentation_feature_manifest_name=classic_segmentation_feature_manifest_name,
        diffae_tracked_feature_manifest_name=diffae_tracked_feature_manifest_name_filtered,
        filter_columns=True,
    )

    return merged_feats_df, merged_feats_df_filtered


def get_and_save_pc_diffae_feats_liveseg_feats_merged_table(dataset_name: str) -> None:
    """Loads the cell-centric DiffAE + segmentation features merged table, computes the PCs, and
    then saves the updated merged table with the PCs as a parquet file.
    """

    out_dir = get_output_path(__file__)

    merged_df_full, merged_df_filtered = get_merged_pc_and_seg_feature_tables(
        dataset_name=dataset_name
    )

    filename = f"{dataset_name}_pc_diffae_seg_feats_merged.parquet"
    merged_df_full.to_parquet(out_dir / filename)

    filename_filtered = f"{dataset_name}_pc_diffae_seg_feats_merged_filtered.parquet"
    merged_df_filtered.to_parquet(out_dir / filename_filtered)


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


def plot_distances_to_fixed_points_for_dataset_multiproc_wrapper(args):
    logger.info(
        "Multiprocessing: plotting distances to fixed points for [ %s ]...",
        args["dataset_name"],
    )
    return plot_distances_to_fixed_points_for_dataset(**args)


def plot_distances_to_fixed_points_for_dataset(
    dataset_name: str,
    min_track_length: int = 216,  # a track duration of 144 is equivalent to 12 hours
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    column_names: list[str] | tuple[str, ...] = DYNAMICS_COLUMN_NAMES,
    out_dir=None,
):
    column_names = list(column_names)

    dataset_config = load_dataset_config(dataset_name)

    if len(dataset_config.shear_stress_regime) > 1:
        logger.warning(
            "Dataset [ %s ] has more than one shear stress condition: [ %s ]. "
            "Skipping for 3D flow field analysis.",
            dataset_name,
            dataset_config.shear_stress_regime,
        )
        return

    if dataset_config.flow_conditions[0].shear_stress == 0:
        logger.warning(
            "Dataset [ %s ] has a shear stress of 0: [ %s ]. "
            "Skipping for 3D flow field analysis.",
            dataset_name,
            dataset_config.flow_conditions[0].shear_stress,
        )
        return

    # If dataset hasn't been processed yet and it has only one
    # flow then make a new output directory for this dataset
    logger.info("Making output directory for dataset [ %s ]...", dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the fixed point dataframe manifest for the given model manifest, run name, and dataset
    base_name = f"{model_manifest_name}_{run_name}_grid"

    fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    # load fixed point dataframe if it exists, and check that required
    # columns are present turn fixed point dataframe into list of arrays of
    # stable fixed point coordinates for each dataset to use for plotting
    fixed_points_dataframe_location = get_dataframe_location_for_dataset(
        fixed_points_dataframe_manifest, dataset_name
    )
    fixed_points_for_dataset = load_dataframe(fixed_points_dataframe_location, delay=False)
    check_required_columns_in_dataframe(
        fixed_points_for_dataset,
        required_columns=[*column_names, Column.DATASET, STABILITY_COLUMN_NAME],
    )

    # if there are no fixed points then move to the next dataset
    if fixed_points_for_dataset.empty:
        logger.warning(
            "No stable fixed points found for dataset [ %s ]. Nothing to plot for this dataset.",
            dataset_name,
        )
        return

    # create a dictionary mapping a fixed point index to its stability
    fp_stability_map = dict(
        zip(
            fixed_points_for_dataset.index,
            fixed_points_for_dataset[STABILITY_COLUMN_NAME],
            strict=True,
        )
    )

    # load the full set of timepoints for the cell-centric data now
    # and do track-specific filtering so that we can see how tracks
    # move in relation to the fixed points over time
    # Load default model manifest and get corresponding feature dataframe
    # manifest name for default run name and specified crop pattern.
    dataframe_manifest_name = (
        "diffae_baseline_exclude_cell_piling_20251110_latent_512_tracked_pca_filtered"
    )
    # load dataframe for the tracked dynamics data
    dataframe_manifest_tracked = load_dataframe_manifest(dataframe_manifest_name)
    df_tracked_delayed = load_dataframe(
        dataframe_manifest_tracked.locations[dataset_name], delay=True
    )
    columns_to_compute = [
        Column.DATASET,
        Column.POSITION,
        Column.TIMEPOINT,
        Column.TRACK_ID,
        Column.CROP_INDEX,
        Column.TRACK_LENGTH,
        *column_names,
    ]
    df_tracked = df_tracked_delayed[columns_to_compute].compute().reset_index(drop=True)

    # determine distance from each fixed point over time and add to the dataframe, along
    # with the signed difference along each axis (e.g. theta, r, rho) from each fixed point
    dist_from_fp_col_prefix = "dist_from_fp_"
    rescaled_theta = PERIOD_THETA_RESCALED + np.pi * (1 - RESCALE_THETA)

    for i in fixed_points_for_dataset.index:
        fpt = fixed_points_for_dataset.iloc[i]

        for col in column_names:
            diff_func = lambda x, fpt=fpt, col=col: (
                np.mod(x - fpt[col] + rescaled_theta / 2, rescaled_theta) - rescaled_theta / 2
                if col == Column.DiffAEData.POLAR_ANGLE.value
                else (x - fpt[col])
            )
            df_tracked[f"diff_from_fp_{col}_{i}"] = diff_func(df_tracked[col])

        dynamics_diff_columns = [f"diff_from_fp_{col}_{i}" for col in column_names]
        df_tracked[f"{dist_from_fp_col_prefix}{i}"] = np.linalg.norm(
            df_tracked[dynamics_diff_columns], axis=1
        )

        dd = (
            df_tracked[f"{dist_from_fp_col_prefix}{i}"]
            .groupby(df_tracked[Column.CROP_INDEX])
            .diff()
        )
        dt = df_tracked[Column.TIMEPOINT].groupby(df_tracked[Column.CROP_INDEX]).diff()
        df_tracked[f"{dist_from_fp_col_prefix}{i}_veloc"] = dd / dt

    # determine which fixed point is closest at each timepoint for each track
    dist_from_fp_columns = [f"{dist_from_fp_col_prefix}{i}" for i in fixed_points_for_dataset.index]
    df_tracked["closest_fp"] = (
        df_tracked[dist_from_fp_columns]
        .idxmin(axis=1)
        .transform(lambda s: int(s.strip(dist_from_fp_col_prefix)))
    )

    # add the stability as a colum name for the closest fixed point at each timepoint
    df_tracked["closest_fp_stability"] = df_tracked["closest_fp"].map(fp_stability_map)

    # filter the data to only include very long tracks
    df_tracked = df_tracked[df_tracked[Column.TRACK_LENGTH] > min_track_length]

    # record how many tracks are included after filtering for long tracks
    num_very_long_tracks = df_tracked[df_tracked[Column.TRACK_LENGTH] > min_track_length][
        Column.TRACK_ID
    ].nunique()
    logger.info(
        "Dataset [ %s ]: %d tracks with duration > %d timepoints.",
        dataset_name,
        num_very_long_tracks,
        min_track_length,
    )

    # plot and save some distances to fixed points
    shear = dataset_config.flow_conditions[0].shear_stress

    # determine if the closest fixed point changes at any timepoint for each track
    df_tracked["closest_fp_changed"] = df_tracked.groupby(["position", "track_id"], as_index=True)[
        "closest_fp"
    ].transform(lambda x: x.diff().fillna(0) != 0)

    # count the number of times the closest fixed point changes for each track
    df_track_fp_switches = (
        df_tracked.groupby(["position", "track_id"], as_index=True)
        .agg(number_of_fp_switches=("closest_fp_changed", "sum"))
        .reset_index()
    )

    # record which closest fixed point each track ends at
    df_tracked["final_closest_fp"] = df_tracked.groupby(["position", "track_id"], as_index=True)[
        "closest_fp"
    ].transform("last")
    final_fp_counts = (
        df_tracked["final_closest_fp"].value_counts(normalize=True) * 100
    ).reset_index(name="percentage")
    final_fp_counts[STABILITY_COLUMN_NAME] = final_fp_counts["final_closest_fp"].map(
        fp_stability_map
    )

    # start plotting desired metrics
    # - distance from fixed point
    # - distance from fixed point along each axis (e.g. theta, r, rho)
    # - velocity towards/away from fixed point
    # - location of fixed points relative to rest of data
    # - how many tracks switch which fixed point is closest at any time
    # - what proportion of tracks finish closest to which fixed points
    fig, ax = plt.subplots()
    ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
    for i in fixed_points_for_dataset.index:
        stability = fixed_points_for_dataset.iloc[i][STABILITY_COLUMN_NAME]
        sns.lineplot(
            df_tracked,
            x=Column.TIMEPOINT,
            y=f"dist_from_fp_{i}",
            ax=ax,
            label=f"FP {i} ({stability})",
        )
    ax.axhline(0, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("distance from fixed point".title())
    ax.set_xlabel("timepoint".title())
    ax.legend(title="fixed point index".title())
    save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp")
    plt.close(fig)

    for i in fixed_points_for_dataset.index:
        stability = fixed_points_for_dataset.iloc[i][STABILITY_COLUMN_NAME]

        fig, ax = plt.subplots()
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        for col in column_names:
            sns.lineplot(
                df_tracked,
                x=Column.TIMEPOINT,
                y=f"diff_from_fp_{col}_{i}",
                alpha=0.5,
                ax=ax,
                label=f"FP {i} ({stability}): {col}",
            )
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("position relative to fixed point along axis".title())
        ax.set_xlabel("timepoint".title())
        ax.legend(title="fixed point index".title())
        save_plot_to_path(fig, out_dir, f"{dataset_name}_signed_dist_from_fp_{i}_components")
        plt.close(fig)

    for i in fixed_points_for_dataset.index:
        lo, hi = np.percentile(df_tracked[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

        fig, ax = plt.subplots()
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        sns.histplot(df_tracked, x=f"dist_from_fp_{i}", y=f"dist_from_fp_{i}_veloc", ax=ax)
        ax.axhline(0, color="red", linestyle="--", alpha=0.7)
        ax.axvline(0, color="grey", linestyle="--", alpha=0.7)
        ax.set_ylim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))

        save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_{i}_veloc")
        plt.close(fig)

    for i in fixed_points_for_dataset.index:
        lo, hi = np.percentile(df_tracked[f"dist_from_fp_{i}_veloc"].dropna(), [1, 99])

        fig, ax = plt.subplots()
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        sns.histplot(df_tracked, x=f"dist_from_fp_{i}_veloc", ax=ax)
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_xlim(-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
        save_plot_to_path(fig, out_dir, f"{dataset_name}_dist_from_fp_{i}_veloc_hist")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(
        data=df_tracked,
        x=Column.DiffAEData.POLAR_ANGLE,
        y=Column.DiffAEData.POLAR_RADIUS,
        color="grey",
        ax=ax,
    )
    sns.scatterplot(
        data=fixed_points_for_dataset,
        x=Column.DiffAEData.POLAR_ANGLE,
        y=Column.DiffAEData.POLAR_RADIUS,
        hue=STABILITY_COLUMN_NAME,
        marker="*",
        palette=STABILITY_COLOR_DICT,
        s=100,
        ax=ax,
    )
    for i, row in fixed_points_for_dataset.iterrows():
        ax.text(
            row[Column.DiffAEData.POLAR_ANGLE],
            row[Column.DiffAEData.POLAR_RADIUS],
            f"FP {i}",
            color=STABILITY_COLOR_DICT.get(row[STABILITY_COLUMN_NAME], "black"),
            fontsize=8,
            ha="right",
            va="bottom",
        )

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, None)
    ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
    ax.set_xlabel(get_label_for_column(Column.DiffAEData.POLAR_ANGLE))
    ax.set_ylabel(get_label_for_column(Column.DiffAEData.POLAR_RADIUS))
    save_plot_to_path(fig, out_dir, f"{dataset_name}_fixed_points_in_polar_space")
    plt.close(fig)

    if df_track_fp_switches["number_of_fp_switches"].any():
        fig, ax = plt.subplots(figsize=(4, 4))
        ax2 = ax.twinx()
        sns.histplot(
            data=df_track_fp_switches,
            x="number_of_fp_switches",
            stat="percent",
            cumulative=True,
            binwidth=1,
            ax=ax,
            color="grey",
            element="step",
            fill=False,
        )
        sns.histplot(
            data=df_track_fp_switches,
            x="number_of_fp_switches",
            stat="percent",
            binwidth=1,
            ax=ax2,
            color="tab:blue",
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("number of fixed point switches per track".title())
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        ax2.set_ylim(0, None)
        ax2.set_ylabel("")
        ax2.spines["right"].set_color("tab:blue")
        ax2.tick_params(axis="y", colors="tab:blue")
        save_plot_to_path(fig, out_dir, f"{dataset_name}_num_fp_switches_hist")
        plt.close(fig)

        sns.histplot(
            data=df_tracked,
            x=Column.TIMEPOINT,
            y="closest_fp_changed",
            # hue=STABILITY_COLUMN_NAME,
            # palette=STABILITY_COLOR_DICT,
        )

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(
            data=final_fp_counts,
            x="final_closest_fp",
            y="percentage",
            hue=STABILITY_COLUMN_NAME,
            ax=ax,
            palette=STABILITY_COLOR_DICT,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("percentage of long tracks".title())
        ax.set_xlabel("final fixed point".title())
        ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        save_plot_to_path(fig, out_dir, f"{dataset_name}_final_fp_stability")
        plt.close(fig)

        # example:
        # dataset 20260302_20X
        # position: 0
        # track id: 585
        # example = df_tracked.query("position==0 and track_id==585")
        # steady_state_start = max(dataset_config.timepoint_annotations["not_steady_state"][0][0])

        # fig, ax = plt.subplots()
        # sns.lineplot(data=example, x="frame_number", y="closest_fp", color="grey", ax=ax, zorder=0)
        # sns.scatterplot(data=example, x="frame_number", y="closest_fp", hue="closest_fp_stability", marker="o", ax=ax)
        # ax.axvline(steady_state_start, c='r', ls=':')
        # ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm², track id: {example['track_id'].iloc[0]}".title())
        # ax.set_xlabel("frame number".title())
        # ax.set_ylabel("closest fixed point".title())
        # save_plot_to_path(fig, out_dir, f"{dataset_name}_track_{example['track_id'].iloc[0]}_closest_fp_over_time")
        # plt.close(fig)

        # fig, ax = plt.subplots()
        # sns.lineplot(data=df_tracked, x="frame_number", y="distance_to_closest_fp", color="grey", ax=ax, zorder=0)
        # sns.scatterplot(data=df_tracked, x="frame_number", y="distance_to_closest_fp", hue="closest_fp", palette="tab20", marker="o", ax=ax)
        # ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        # ax.set_xlabel("frame number".title())
        # ax.set_ylabel("distance to closest fixed point".title())
        # save_plot_to_path(fig, out_dir, f"{dataset_name}_distance_to_closest_fp_over_time")
        # plt.close(fig)

        # fig, ax = plt.subplots()
        # for i in fixed_points_for_dataset.index:
        #     stability = fixed_points_for_dataset.iloc[i][STABILITY_COLUMN_NAME]
        #     if stability == "stable":
        #         # line_style = "-"
        #         pass
        #     else:
        #         # line_style = "--"
        #         continue

        #     sns.lineplot(
        #         example,
        #         x=Column.TIMEPOINT,
        #         y=f"dist_from_fp_{i}",
        #         # ls=line_style,
        #         ax=ax,
        #         label=f"FP {i} ({stability})",
        #     )
        # ax.set_ylim(0)
        # ax.set_title(f"{dataset_name}, shear stress: {shear} dyn/cm²".title())
        # ax.set_xlabel("frame number".title())
        # ax.set_ylabel("distance from fixed point".title())
        # save_plot_to_path(fig, out_dir, f"{dataset_name}_distance_from_fp_over_time")
        # plt.close(fig)
