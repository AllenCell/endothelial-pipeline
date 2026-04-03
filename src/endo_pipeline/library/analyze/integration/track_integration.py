import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import color_palette

from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.data_driven_flow_field import solve_ddff_ode
from endo_pipeline.library.analyze.dataframe_validation import check_required_columns_in_dataframe
from endo_pipeline.library.analyze.kramers_moyal.km_computation import get_kramers_moyal_coeffs
from endo_pipeline.library.analyze.kramers_moyal.km_kernels import KramersMoyalKernel
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.analyze.numerics.forward_difference import get_traj_and_diff
from endo_pipeline.library.analyze.optical_flow_calculator import one_direction_vector_field_example
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import (
    DIFFAE_PC_COLUMN_NAMES,
    NUM_PCS_TO_ANALYZE,
)
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    DYNAMICS_COLUMN_NAMES,
    KERNEL_BANDWIDTHS_DYNAMICS,
    KERNEL_NAMES_DYNAMICS,
    PERIOD_THETA_RESCALED,
    RESCALE_THETA,
)
from endo_pipeline.settings.flow_field_3d import (
    BIN_WIDTH_DEFAULTS,
    INIT_POINT_3D,
    TIME_STEP_IN_MINUTES,
    TRAJECTORY_TIME_SPAN,
)
from endo_pipeline.settings.flow_field_dataframes import (
    DATAFRAME_MANIFEST_PREFIX_DRIFT,
    DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS,
    STABILITY_COLUMN_NAME,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_COLUMNS_TO_DROP,
    DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_FILTERED,
    DEFAULT_DIFFAE_PCA_FEATURE_TRACKED_MANIFEST_NAME_UNFILTERED,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_SEG_FEATURE_MANIFEST_NAME,
)

logger = logging.getLogger(__name__)


def get_flow_field_estimation_kernels(
    column_names: list[str | Column.DiffAEData] | None = None,
    rescale_theta: bool = RESCALE_THETA,
    period_theta_rescaled: float = PERIOD_THETA_RESCALED,
    kernel_names_dynamics: dict[Column.DiffAEData, str] = KERNEL_NAMES_DYNAMICS,
    kernel_bandwidths_dynamics: dict[Column.DiffAEData, float] = KERNEL_BANDWIDTHS_DYNAMICS,
) -> list[KramersMoyalKernel]:
    """Return the kernels used for flow field estimation for the specified columns."""
    # initialize kernels for each of the three variables for flow field estimation
    kernels: list[KramersMoyalKernel] = []
    rescaled_theta = period_theta_rescaled + np.pi * (1 - rescale_theta)

    # Get the corresponding kernels for each variable. For the polar angle variable,
    # also specify the period for the kernel based on the rescaled theta range, to
    # ensure that the periodicity of the polar angle is taken into account in the
    # flow field estimation.
    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    for column_name in column_names:
        name = kernel_names_dynamics[column_name]
        bandwidth = kernel_bandwidths_dynamics[column_name]
        period = rescaled_theta if column_name == Column.DiffAEData.POLAR_ANGLE else None
        kernels.append(KramersMoyalKernel(name=name, bandwidth=bandwidth, period=period))
    return kernels


def get_flow_field_estimation_bin_widths(
    column_names: list[str | Column.DiffAEData] | None = None,
    bin_widths_dynamics: dict[Column.DiffAEData, float] = BIN_WIDTHS_DYNAMICS,
) -> list[float]:
    """Return the bin widths used for flow field estimation for the specified columns."""
    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    bin_widths: list[float] = []
    for column_name in column_names:
        bin_width = bin_widths_dynamics[column_name]
        bin_widths.append(bin_width)
    return bin_widths


# def process_dataset_for_track_integration(
#     dataset_name: str,
#     merged_cellcentric_features_manifest_name: str = DEFAULT_PC_DIFFAE_SEG_FEATURE_MANIFEST_NAME_FILTERED,
#     diffae_grid_manifest_name: str = DEFAULT_DIFFAE_PCA_FEATURE_GRID_MANIFEST_NAME_FILTERED,
#     make_integrated_plots: bool = True,
#     dynamics_columns: list[Column.DiffAEData] = list(DYNAMICS_COLUMN_NAMES),
# ) -> None:
#     logger.info("Processing dataset: [ %s ]", dataset_name)

#     # set workflow defaults
#     out_subdir = get_output_path(__file__, dataset_name)
#     # column_names = list(DYNAMICS_COLUMN_NAMES)  # dynamics_column_names = theta, r, rho

#     # load the track-based diffae + segmentation feature merged manifest
#     # and the grid-based diffae manifest
#     diffae_tracked_manifest = load_dataframe_manifest(merged_cellcentric_features_manifest_name)
#     diffae_tracked_location = get_dataframe_location_for_dataset(
#         diffae_tracked_manifest, dataset_name
#     )
#     diffae_tracked_df_delayed = load_dataframe(diffae_tracked_location, delay=True)

#     diffae_grid_manifest = load_dataframe_manifest(diffae_grid_manifest_name)
#     diffae_grid_location = get_dataframe_location_for_dataset(diffae_grid_manifest, dataset_name)
#     diffae_grid_df_delayed = load_dataframe(diffae_grid_location, delay=True)

#     # keep only the columns that are needed for the analysis to reduce memory usage
#     cols_to_keep = [
#         Column.DATASET,
#         Column.POSITION,
#         Column.TIMEPOINT,
#         Column.TRACK_ID,
#         Column.SegData.LABEL,
#         Column.CROP_INDEX,
#         Column.DiffAEData.MODEL_MANIFEST,
#         Column.SegData.TIME_HRS,
#         Column.SegData.TIME_MINS,
#         Column.TRACK_LENGTH,
#     ] + list(dynamics_columns)
#     diffae_tracked_df = diffae_tracked_df_delayed[cols_to_keep].compute()
#     diffae_grid_df = diffae_grid_df_delayed[cols_to_keep].compute()

#     # load or compute the trajectories and flow fields for the grid-based
#     # and cell-centric crops
#     flow_field_dict_grids, fixed_points_df = get_flow_field_and_fixed_points(
#         dataset_name=dataset_name,
#         column_names=dynamics_columns,
#         model_manifest_name=DEFAULT_MODEL_MANIFEST_NAME,
#         run_name=DEFAULT_MODEL_RUN_NAME,
#     )

#     for i, fp_row in fixed_points_df.iterrows():
#         flow_field_slices = (
#             fp_row[dynamics_columns[2]],
#             fp_row[dynamics_columns[1]],
#         )  # feature 3, feature 2
#         fixed_points_at_slices = (
#             fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[2]]),
#             fp_row[list(map(str, dynamics_columns))].drop(index=[dynamics_columns[1]]),
#         )
#     # # get the slice indexes to use for plotting the flow fields
#     # # (we will be setting PC3 to a constant, i.e. the z-axis here)
#     # _, slice_indexes = get_valid_slice_indexes(diffae_grid_df, traj_grids, flow_field_dict_grids)

#     # get flow field vectors and grid points to plot
#     v1_grids, v2_grids, v3_grids = flow_field_dict_grids["vectors"]
#     g1_grids, g2_grids, g3_grids = flow_field_dict_grids["grid"]
#     v1_tracks, v2_tracks, v3_tracks = flow_field_dict_tracks["vectors"]
#     g1_tracks, g2_tracks, g3_tracks = flow_field_dict_tracks["grid"]

#     # Plot the quiver slices for the grid-based and cell-centric crops
#     # at the full resolution:
#     out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_full_quiver.png"
#     fig, ax = plot_grid_vs_tracks_flow_field(
#         v1_grids,
#         v2_grids,
#         g1_grids,
#         g2_grids,
#         v1_tracks,
#         v2_tracks,
#         g1_tracks,
#         g2_tracks,
#         slice_indexes=slice_indexes,
#         ds=1,
#         scale=60,
#     )
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # Plot the quiver slices for the grid-based and cell-centric crops
#     # at the standard/default resolution and include the fixed points
#     # for both the grid and cell-centric crops:
#     out_path = out_subdir / f"{dataset_name}_quiver_slice_comparison_partial_quiver.png"
#     fig, ax = plot_grid_vs_tracks_flow_field(
#         v1_grids,
#         v2_grids,
#         g1_grids,
#         g2_grids,
#         v1_tracks,
#         v2_tracks,
#         g1_tracks,
#         g2_tracks,
#         slice_indexes=slice_indexes,
#     )
#     # add the grid crop based fixed point from the trajectory:
#     ax.scatter(
#         traj_grids[-1, 0],
#         traj_grids[-1, 1],
#         s=250,
#         color="cyan",
#         marker="*",
#         lw=1,
#         edgecolor="darkblue",
#         zorder=10,
#     )
#     # add the cell-centric crop based fixed point from the trajectory:
#     ax.scatter(
#         traj_tracks[-1, 0],
#         traj_tracks[-1, 1],
#         s=250,
#         color="yellow",
#         marker="*",
#         lw=1,
#         edgecolor="darkred",
#         zorder=10,
#     )
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     fig.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     # Plot the angular deviation between the grid and cell-centric crop-based
#     # flow field vectors:
#     angles = get_vector_angles_as_grid(
#         v1_grids,
#         v2_grids,
#         v3_grids,
#         v1_tracks,
#         v2_tracks,
#         v3_tracks,
#         slice_indexes,
#     )
#     grid_vs_track_vec_angle_hist2d(
#         angles,
#         out_subdir,
#         filename=f"{dataset_name}_vecvec_angles",
#         extent=(*ax.get_xlim(), *ax.get_ylim()),
#     )

#     # Plot the dot product between the grid and cell-centric crop-based
#     dot_prod = get_vector_dot_products_as_grid(
#         v1_grids,
#         v2_grids,
#         v3_grids,
#         v1_tracks,
#         v2_tracks,
#         v3_tracks,
#         slice_indexes,
#     )
#     grid_vs_track_vec_dot_prod_hist2d(
#         dot_prod,
#         out_subdir,
#         filename=f"{dataset_name}_vecvec_dot_products",
#         extent=(*ax.get_xlim(), *ax.get_ylim()),
#     )

#     # Compare the angles between grid crop PC vectors
#     # and the PC vectors of a single track
#     diffae_tracked_df["dpc1"] = diffae_tracked_df.groupby(Column.CROP_INDEX)["pc_1"].diff()
#     diffae_tracked_df["dpc2"] = diffae_tracked_df.groupby(Column.CROP_INDEX)["pc_2"].diff()
#     diffae_tracked_df["dt"] = diffae_tracked_df.groupby(Column.CROP_INDEX)[
#         Column.SegData.TIME_MINS
#     ].diff()

#     # create partial functions from get_approx_point_from_grid to pass
#     # along to the groupby.apply() method
#     get_approx_grid_bin = lambda pc1_pc2_arr: get_approx_point_from_grid(
#         pc1_pc2_arr,
#         g1_grids,
#         g2_grids,
#         v1_grids,
#         v2_grids,
#         slice_indexes,
#     )
#     get_approx_grid_bin_from_df = lambda df: pd.DataFrame(
#         columns=[["pc_1", "pc_2"]], data=get_approx_grid_bin(df.to_numpy()), index=df.index
#     )

#     get_approx_grid_vec = lambda pc1_pc2_arr: get_approx_vec_from_grid(
#         pc1_pc2_arr,
#         g1_grids,
#         g2_grids,
#         v1_grids,
#         v2_grids,
#         slice_indexes,
#     )
#     get_approx_grid_vec_from_df = lambda df: pd.DataFrame(
#         columns=[["pc_1", "pc_2"]], data=get_approx_grid_vec(df.to_numpy()), index=df.index
#     )

#     # Apply the partial functions to the DataFrame to get the approximate grid bin
#     # and vector associated with each cell-centric PC1 and PC2 value
#     diffae_tracked_df[["approx_bin_pc1", "approx_bin_pc2"]] = (
#         diffae_tracked_df.groupby(Column.DATASET, as_index=False)
#         .apply(lambda df: get_approx_grid_bin_from_df(df[["pc_1", "pc_2"]]))
#         .droplevel(level=0)
#     )
#     diffae_tracked_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
#         diffae_tracked_df.groupby(Column.CROP_INDEX, as_index=False)
#         .apply(lambda df: get_approx_grid_vec_from_df(df[["pc_1", "pc_2"]]))
#         .droplevel(level=0)
#     )

#     # Compute the angle between the approximate grid vector
#     # and the the vector from the cell-centric PC1 and PC2
#     # both in radians and degrees
#     diffae_tracked_df["track_angle_deviation_rad"] = get_vector_vector_angle_fast(
#         diffae_tracked_df[["approx_vec_pc1", "approx_vec_pc2"]].values,
#         diffae_tracked_df[["dpc1", "dpc2"]].values,
#     )
#     diffae_tracked_df["track_angular_deviation_deg"] = diffae_tracked_df[
#         "track_angle_deviation_rad"
#     ].transform(np.rad2deg)

#     diffae_tracked_df["pc1_pc2_vec_mag"] = np.linalg.norm(
#         diffae_tracked_df[["dpc1", "dpc2"]].values, axis=1
#     )

#     # group dataframe by a combination of dataset, position, and crop index
#     # note that we have replaced the track id with the crop index in this
#     # case because the crop index is unique throughout all 6 positions,
#     # whereas the track id is only unique within a single position
#     mean_track_deviation_dfs = (
#         diffae_tracked_df.groupby(["dataset_name", "position_as_str", "crop_index"])[
#             ["track_angular_deviation_deg", "pc1_pc2_vec_mag"]
#         ]
#         .agg("mean")
#         .reset_index()
#     )

#     plot_and_save_track_flow_field_deviations(
#         mean_track_deviation_dfs=mean_track_deviation_dfs,
#         out_subdir=out_subdir,
#         dataset_name=dataset_name,
#     )

#     # get the dot products
#     diffae_tracked_df["dot_product_grid_vs_cell"] = np.einsum(
#         "ij,ij->i",
#         diffae_tracked_df[["approx_vec_pc1", "approx_vec_pc2"]],
#         diffae_tracked_df[["dpc1", "dpc2"]],
#     )
#     # also aggregate the dot products by crop index (i.e. unique track id across all positions)
#     diffae_tracked_dot_prod_agg = (
#         diffae_tracked_df.groupby("crop_index")["dot_product_grid_vs_cell"]
#         .agg(["mean", "median"])
#         .reset_index()
#     )

#     plot_title = "Mean per track"
#     col_name = "mean"
#     plot_and_save_track_flow_field_dot_product_histogram(
#         features_dataframe=diffae_tracked_dot_prod_agg,
#         feature_column_name=col_name,
#         out_dir=out_subdir,
#         filename=f"{dataset_name}_dot_product_grid_vs_cell_{col_name}",
#         plot_title=plot_title,
#     )

#     plot_title = "Non-aggregated dot products"
#     col_name = "dot_product_grid_vs_cell"
#     plot_and_save_track_flow_field_dot_product_histogram(
#         features_dataframe=diffae_tracked_df,
#         feature_column_name=col_name,
#         out_dir=out_subdir,
#         filename=f"{dataset_name}_dot_product_grid_vs_cell_{col_name}",
#         plot_title=plot_title,
#     )

#     if make_integrated_plots:
#         # NOTE: this is a very memory-intensive operation despite my attempts to
#         # reduce memory needs here, so if you change the minimum track duration
#         # then expect the workflow to require a lot more memory or crash if you
#         # don't have enough
#         diffae_tracked_df = diffae_tracked_df.query("track_duration > 180")
#         groups = diffae_tracked_df.groupby([Column.DATASET, Column.POSITION, Column.CROP_INDEX])

#         i = 0
#         for nm, df in tqdm(groups, desc=dataset_name):
#             ds_nm, pos, tid = nm
#             assert (
#                 tid % 1
#             ) == 0, f"Track ID should be an integer or convertible to an integer. Got {tid}."
#             hue_min = -1 * np.nanmax(diffae_tracked_df["dot_product_grid_vs_cell"].abs())
#             hue_max = 1 * np.nanmax(diffae_tracked_df["dot_product_grid_vs_cell"].abs())
#             hue_center = 0.0
#             plot_pc_integrated_track_as_arrows(
#                 dataset_name=str(ds_nm),
#                 position_name=str(pos),
#                 track_id=int(tid),
#                 df=df,
#                 v1_grids=v1_grids,
#                 v2_grids=v2_grids,
#                 g1_grids=g1_grids,
#                 g2_grids=g2_grids,
#                 slice_indexes=slice_indexes,
#                 out_subdir=out_subdir,
#                 hue_min=hue_min,
#                 hue_max=hue_max,
#                 hue_center=hue_center,
#                 cmap_name="managua",
#                 hued_feat_name="dot_product_grid_vs_cell",
#                 track_alpha=0.5,
#             )
#             i += 1
#             if i % 100 == 0:
#                 # force garbage collection to keep memory free when
#                 # creating plots from a loop every 100th iteration
#                 gc.collect()

#     # overlay flow fields on the histograms of the data to see where
#     # most of the data being used to extrapolate flow fields is
#     overlay_flow_fields_on_histograms(
#         dataset_name,
#         out_subdir,
#         diffae_grid_df,
#         diffae_tracked_df,
#         v1_grids,
#         v2_grids,
#         g1_grids,
#         g2_grids,
#         v1_tracks,
#         v2_tracks,
#         g1_tracks,
#         g2_tracks,
#         slice_indexes,
#     )


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
    pc_cols_to_drop = DIFFAE_PC_COLUMN_NAMES[100:]
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
    column_names: list[str | Column.DiffAEData] | None = None,
    load_precomputed_trajectories: Path | None = None,
) -> tuple[np.ndarray, dict]:

    if column_names is None:
        column_names = list(DYNAMICS_COLUMN_NAMES)

    # set kernel params
    kernels = get_flow_field_estimation_kernels(column_names)

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
    traj_list, d_traj_list = get_traj_and_diff(df, list(column_names))

    # get drift and diffusion estimates
    # (Kramers-Moyal coefficients)
    drift_km, diff_km = get_kramers_moyal_coeffs(
        traj_list, d_traj_list, bins=bins, dt=dt, kernel=kernels
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


def get_flow_field_and_fixed_points(
    dataset_name: str,
    column_names: list[str | Column.DiffAEData] | None = None,
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
) -> tuple[dict, pd.DataFrame]:
    """
    Return the flow fields and fixed points for the grid-based crops by loading them from the
    corresponding dataframe manifests for the given dataset, model and run name.
    The flow field dictionaries are constructed from the drift data in the drift dataframe manifest
    and the fixed points are loaded from the fixed points dataframe manifest for the given dataset.

    Parameters
    ----------
    dataset_name
        Name of the dataset for which to load the flow field and fixed points.
    column_names
        List of column names corresponding to the dynamics features to use for constructing the flow field,
        by default None
    model_manifest_name
        Name of the model dataframe manifest to use for loading the drift data, by default DEFAULT_MODEL_MANIFEST_NAME
    run_name
        Name of the model run to use for loading the drift data, by default DEFAULT_MODEL_RUN_NAME

    Returns
    -------
    :
        The flow field dictionary and the fixed points dataframe for the given dataset.

    """

    base_name = f"{model_manifest_name}_{run_name}_grid"
    fixed_points_df_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
    fixed_points_df_manifest = load_dataframe_manifest(fixed_points_df_manifest_name)

    drift_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_DRIFT}_{base_name}"
    drift_dataframe_manifest = load_dataframe_manifest(drift_dataframe_manifest_name)

    if dataset_name not in fixed_points_df_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in fixed points dataframe manifest [ %s ]!",
            dataset_name,
            fixed_points_df_manifest_name,
        )
        return {}, pd.DataFrame()
    if dataset_name not in drift_dataframe_manifest.locations:
        logger.warning(
            "Dataset [ %s ] not found in drift dataframe manifest [ %s ]!",
            dataset_name,
            drift_dataframe_manifest_name,
        )
        return {}, pd.DataFrame()

    logger.info("Getting flow fields and fixed points for grid-based crops...")

    # load fixed point dataframe if it exists, and check that required
    # columns are present turn fixed point dataframe into list of arrays of
    # stable fixed point coordinates for each dataset to use for plotting
    fixed_points_df_location = get_dataframe_location_for_dataset(
        fixed_points_df_manifest, dataset_name
    )
    fixed_points_df = load_dataframe(fixed_points_df_location, delay=False)
    check_required_columns_in_dataframe(
        fixed_points_df,
        required_columns=[*column_names, Column.DATASET, STABILITY_COLUMN_NAME],
    )

    # if there are no fixed points then move to the next dataset
    if fixed_points_df.empty:
        logger.warning(
            "No stable fixed points found for dataset [ %s ]. Nothing to plot for this dataset.",
            dataset_name,
        )

    drift_dataframe_location = get_dataframe_location_for_dataset(
        drift_dataframe_manifest, dataset_name
    )
    drift_df = load_dataframe(drift_dataframe_location, delay=False)

    # restructure the drift dataframe into a flow field dictionary
    ndim = len(column_names)
    drift_column_names = [f"{name}_drift" for name in column_names]

    grid_points_1d = [np.sort(drift_df[column_name].unique()) for column_name in column_names]
    grid_shape = tuple(len(points) for points in grid_points_1d)
    grid = np.meshgrid(*grid_points_1d, indexing="ij")

    drift_values = drift_df[drift_column_names].to_numpy().reshape(*grid_shape, ndim)
    drift_vector_field = [drift_values[..., i] for i in range(ndim)]
    flow_field_dict = {"vectors": drift_vector_field, "grid": grid}

    return flow_field_dict, fixed_points_df


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
