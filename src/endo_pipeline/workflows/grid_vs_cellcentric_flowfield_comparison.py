import gc
import logging

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.configs.dataset_io import ipython_cli_flexecute
from endo_pipeline.io import configure_logging, get_output_path
from endo_pipeline.library.analyze.integration.track_integration import (
    get_approx_point_from_grid,
    get_approx_vec_from_grid,
    get_gridcrop_and_cellcentric_trajectories_and_flow_fields,
    get_preprocessed_manifests_and_km_bounds,
    get_vector_angles_as_grid,
    get_vector_dot_products_as_grid,
    get_vector_vector_angle_fast,
    make_angular_deviation_test,
)
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
from endo_pipeline.manifests import load_model_manifest
from endo_pipeline.settings import DEFAULT_SEG_FEATURE_MANIFEST_NAME

logger = logging.getLogger(__name__)

# the below 2 lines are both used to control memory
# usage problems when making many plots in a loop
matplotlib.use("Agg")
plt.ioff()  # turns off interactive mode in matplotlib


def process_dataset(
    dataset_name: str,
    datasets_for_bounds: list[str],
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
    make_integrated_plots: bool = True,
) -> None:
    logger.info(f"Processing dataset: {dataset_name}")

    out_subdir = get_output_path(__file__, dataset_name, include_timestamp=False)
    configure_logging(out_subdir, logger, verbose=True)

    model_manifest = load_model_manifest(model_manifest_name)

    # load and preprocess the different diffae manifests and PCA pipeline
    merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
        dataset_name=dataset_name,
        model_manifest=model_manifest,
        run_name=run_name,
        seg_feature_manifest_name=seg_feature_manifest_name,
        datasets_for_bounds=datasets_for_bounds,
    )

    # keep only the columns that are needed for the analysis to reduce memory usage
    cols_to_keep = [
        "dataset_name",
        "position",
        "position_as_str",
        "track_id",
        "label",
        "crop_index",
        "model_manifest_name",
        "model_name",
        "image_index",
        "frame_number",
        "time_hours",
        "time_minutes",
        "track_duration",
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
    merged_feats_df["dpc1"] = merged_feats_df.groupby("crop_index")["pc_1"].diff()
    merged_feats_df["dpc2"] = merged_feats_df.groupby("crop_index")["pc_2"].diff()
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
        merged_feats_df.groupby("crop_index", as_index=False)
        .apply(lambda df: get_approx_grid_bin_from_df(df[["pc_1", "pc_2"]]))
        .droplevel(level=0)
    )
    merged_feats_df[["approx_vec_pc1", "approx_vec_pc2"]] = (
        merged_feats_df.groupby("crop_index", as_index=False)
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
        groups = merged_feats_df.groupby(["dataset_name", "position_as_str", "crop_index"])

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

    return


def main(
    dataset_collection_name: str = "pca_reference_legacy",
    model_manifest_name: str = "diffae_04_10",
    run_name: str | None = None,
    seg_feature_manifest_name: str = DEFAULT_SEG_FEATURE_MANIFEST_NAME,
) -> None:
    """
    Makes plots comparing cell-centric and grid-based flow fields.
    """

    dataset_name_list = get_datasets_in_collection(dataset_collection_name)

    for dataset_name in dataset_name_list:
        logger.info(f"Processing {dataset_name}...")
        process_dataset(
            dataset_name=dataset_name,
            model_manifest_name=model_manifest_name,
            run_name=run_name,
            seg_feature_manifest_name=seg_feature_manifest_name,
            datasets_for_bounds=dataset_name_list,
            make_integrated_plots=True,
        )

    # create a test flow field and test set of vectors
    # to check that the angular deviation calculation
    # works as expected
    out_dir = get_output_path(__file__, include_timestamp=False)
    make_angular_deviation_test(out_dir)


if __name__ == "__main__":
    ipython_cli_flexecute(main)
