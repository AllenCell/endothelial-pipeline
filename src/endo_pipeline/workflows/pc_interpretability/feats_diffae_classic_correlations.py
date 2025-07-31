import re
from pathlib import Path

import numpy as np
import pandas as pd

# for data exploration; remove later
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from skimage.measure import regionprops

from src.endo_pipeline.configs import load_dataset_collection_config
from src.endo_pipeline.configs.dataset_io import load_nuclei_prediction
from src.endo_pipeline.io import get_output_path
from src.endo_pipeline.library.analyze.integration.track_integration import (  # get_approx_point_from_grid,; get_approx_vec_from_grid,; get_gridcrop_and_cellcentric_trajectories_and_flow_fields,; get_vector_angles_as_grid,; get_vector_dot_products_as_grid,; get_vector_vector_angle_fast,; make_angular_deviation_test,
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.library.process.general_image_preprocessing import get_default_dim_order

if __name__ == "__main__":

    dataset_name_list = load_dataset_collection_config("pca_reference").datasets
    dataset_name = dataset_name_list[0]  # for testing purposes
    out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

    # load and preprocess the different diffae manifests and PCA pipeline
    # NOTE: this takes a little over a minute to load; we can consider
    # using dask dataframes and only computing the desired columns
    merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
        dataset_name, datasets_for_bounds=dataset_name_list
    )

    # # keep only the columns that are needed for the analysis to reduce memory usage
    # cols_to_keep = [
    #     "dataset_name",
    #     "position",
    #     "position_as_str",
    #     "track_id",
    #     "label",
    #     "crop_index",
    #     "mlflow_id",
    #     "model_name",
    #     "image_index",
    #     "frame_number",
    #     "time_hours",
    #     "time_minutes",
    #     "track_duration",
    # ] + [col for col in merged_feats_df.columns if "feat" in col or "pc" in col]
    # merged_feats_df = merged_feats_df[cols_to_keep]

    pc_col_names = [col_nm for col_nm in merged_feats_df.columns if re.match("pc[0-9]", col_nm)]
    feat_col_names = [
        col_nm for col_nm in merged_feats_df.columns if re.match("feat_[0-9]+", col_nm)
    ]
    measured_col_names = [
        "alignment_deg_rel_to_flow",
        "area",
        "perimeter",
        "eccentricity",
        "aspect_ratio",
        "nematic_order",
        "centroid_Y",
        "centroid_X",
        "nuc_with_most_overlap_0_centroid_Y",
        "nuc_with_most_overlap_0_centroid_X",
        "cell_fluorescence_mean (a.u.)",
        "cell_solidity",
        "number_of_neighbors",
        "nuc_pos_rel_cell_angle_deg",
    ]
    for col in measured_col_names + pc_col_names:
        if col not in merged_feats_df.columns:
            print(
                f"Column {col} not found in merged_feats_df. Available columns: {merged_feats_df.columns.tolist()}"
            )

    for pc in pc_col_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{pc.upper()}")
        sns.lineplot(data=merged_feats_df, x="time_hours", y=pc)
        # ax2 = ax.twinx()
        # sns.lineplot(
        #     data=merged_feats_df,
        #     x="time_hours",
        #     y="alignment_deg_rel_to_flow",
        #     ax=ax2,
        #     c="tab:orange",
        # )
        plt.show()

    for prop in measured_col_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{prop}")
        sns.lineplot(data=merged_feats_df, x="time_hours", y=prop)
        # ax2 = ax.twinx()
        # sns.lineplot(
        #     data=merged_feats_df,
        #     x="time_hours",
        #     y="alignment_deg_rel_to_flow",
        #     ax=ax2,
        #     c="tab:orange",
        # )
        plt.show()

    # compute the correlation between the PCs and the alignment to flow
    correlation_results = []
    for pc in pc_col_names:
        result = pearsonr(
            merged_feats_df[pc],
            merged_feats_df["alignment_deg_rel_to_flow"],
        )
        correlation_results.append(result)

        # print(f"PC{i} vs. Alignment to Flow: r={corr:.2f}, p={pval:.2e}")


def get_num_nuc_centroids_in_crop(merged_feats_df: pd.DataFrame) -> None:  # pd.Series:
    """
    Get the number of nuclei centroids in each crop.
    """

    # Get the columns that contain the nuclei centroids
    nuclei_centroid_Y_cols = [
        col
        for col in merged_feats_df.columns
        if "nuc_with_most_overlap" in col and "centroid_Y" in col
    ]
    nuclei_centroid_X_cols = [
        col
        for col in merged_feats_df.columns
        if "nuc_with_most_overlap" in col and "centroid_X" in col
    ]
    # Make sure that there are the same number of Y and X centroid columns
    assert len(nuclei_centroid_Y_cols) == len(
        nuclei_centroid_X_cols
    ), f"Mismatch in number of Y and X centroid columns (Y: {len(nuclei_centroid_Y_cols)}, X: {len(nuclei_centroid_X_cols)})."
    # Make sure that each centroid column has a corresponding X and Y dimension in the same position
    for i in range(len(nuclei_centroid_Y_cols)):
        assert (
            nuclei_centroid_Y_cols[i].replace("centroid_Y", "centroid_X")
            == nuclei_centroid_X_cols[i]
        ), f"Mismatch in Y and X centroid column names: {nuclei_centroid_Y_cols[i]} vs {nuclei_centroid_X_cols[i]}."

    # Collapse the nuclei centroid columns into a single 1D array for each dimension
    # Also adjust the centroid positions based on the resolution level in this step
    nuclei_centroids_Y = (
        merged_feats_df[nuclei_centroid_Y_cols].values
        // (merged_feats_df[["resolution_level"]] + 1).values
    ).flatten()
    nuclei_centroids_X = (
        merged_feats_df[nuclei_centroid_X_cols].values
        // (merged_feats_df[["resolution_level"]] + 1).values
    ).flatten()

    num_is_between = lambda x, start, end: (x >= start) & (x <= end)

    test = np.logical_and(
        num_is_between(
            merged_feats_df["nuc_with_most_overlap_0_centroid_Y"]
            // (merged_feats_df["resolution_level"] + 1),
            merged_feats_df["start_y"],
            merged_feats_df["end_y"],
        ),
        num_is_between(
            merged_feats_df["nuc_with_most_overlap_0_centroid_X"]
            // (merged_feats_df["resolution_level"] + 1),
            merged_feats_df["start_x"],
            merged_feats_df["end_x"],
        ),
    )

    # return


# def get_num_nuclei_in_array(np.ndarray) -> int:


#     return num_nuclei

# def get_num_nuclei_in_crop(bounds_Y, bounds_X, method: Literal["centroid", "any"]) -> int:


# ensure the crop coordinates have an integer datatype
merged_feats_df["start_y"] = (
    merged_feats_df["start_y"] * (merged_feats_df["resolution_level"] + 1)
).astype(int)
merged_feats_df["end_y"] = (
    merged_feats_df["end_y"] * (merged_feats_df["resolution_level"] + 1)
).astype(int)
merged_feats_df["start_x"] = (
    merged_feats_df["start_x"] * (merged_feats_df["resolution_level"] + 1)
).astype(int)
merged_feats_df["end_x"] = (
    merged_feats_df["end_x"] * (merged_feats_df["resolution_level"] + 1)
).astype(int)

dim_order = get_default_dim_order()
groups = merged_feats_df.groupby(["dataset_name", "position", "image_index"])
for (dataset_name, position, timeframe), df in groups:
    # get the nuclei prediction
    nuc_seg = load_nuclei_prediction(
        dataset_name=dataset_name,
        position=position,
        T=timeframe,
        dim_order=dim_order,
    )
    nuc_seg_arr = nuc_seg.compute()

    props = regionprops(nuc_seg_arr.squeeze())
    # find how many centroids nuclei are found in the crop coordinate
    dim_shapes = dict(zip(dim_order, nuc_seg.shape))
    dim_order_squeezed = "".join([d for d in dim_order if dim_shapes[d] > 1])
    # nuc_centroids_Y = []
    # nuc_centroids_X = []
    # for p in props:
    #     nuc_centroids_Y.append(p.centroid[dim_order_squeezed.index("Y")])
    #     nuc_centroids_X.append(p.centroid[dim_order_squeezed.index("X")])

    # centroids
    # the ndmin=2 is so that the p.centroid shape is the same as p.coords
    centroid_indices_Y = []
    centroid_indices_X = []
    for p in props:
        # get the nuclei centroid coordinates
        coords = np.array(p.centroid, ndmin=2).astype(float)
        centroid_indices_Y.append(coords[..., dim_order_squeezed.index("Y")])
        centroid_indices_X.append(coords[..., dim_order_squeezed.index("X")])

    # centroid_indices = [np.array(p.centroid, ndmin=2) for p in props]
    centroid_indices_Y = np.stack(centroid_indices_Y)
    centroid_indices_X = np.stack(centroid_indices_X)

    # any
    # find the largest nuclei in the image
    biggest_nuc_mask = max([p.num_pixels for p in props])
    # and use it's shape to pad any nuclei coordinates so that
    # the arrays of coordinates all have the same shape
    whole_mask_indices_Y = []
    whole_mask_indices_X = []
    for p in props:
        # get the nuclei coordinates
        coords = p.coords.astype(float)
        # define how much padding you need to add to these nuclei coordinates
        pad_width = ((0, biggest_nuc_mask - p.coords.shape[0]), (0, 0))
        # do the padding
        coords = np.pad(coords, pad_width, mode="constant", constant_values=np.nan)
        # add the padded coordinates to the list
        whole_mask_indices_Y.append(coords[..., dim_order_squeezed.index("Y")])
        whole_mask_indices_X.append(coords[..., dim_order_squeezed.index("X")])

    whole_mask_indices_Y_arr = np.stack(whole_mask_indices_Y)
    whole_mask_indices_X_arr = np.stack(whole_mask_indices_X)

    test_Y = np.logical_and(
        np.reshape(whole_mask_indices_Y, (*whole_mask_indices_Y_arr.shape, 1))
        >= np.reshape(df["start_y"], (1, 1, len(df["start_y"]))),
        np.reshape(whole_mask_indices_Y, (*whole_mask_indices_X_arr.shape, 1))
        <= np.reshape(df["end_y"], (1, 1, len(df["end_y"]))),
    )
    test_Y = test_Y.any(axis=1)

    test_X = np.logical_and(
        np.reshape(whole_mask_indices_X, (*whole_mask_indices_Y_arr.shape, 1))
        >= np.reshape(df["start_x"], (1, 1, len(df["start_x"]))),
        np.reshape(whole_mask_indices_X, (*whole_mask_indices_X_arr.shape, 1))
        <= np.reshape(df["end_x"], (1, 1, len(df["end_x"]))),
    )
    test_X = test_X.any(axis=1)

    num_nuclei_in_crop = np.logical_and(test_Y, test_X).sum(axis=0)

    # for getting number of nuclei from arbitrary array
    test_slice = (
        slice(df["start_y"].iloc[0], df["end_y"].iloc[0]),
        slice(df["start_x"].iloc[0], df["end_x"].iloc[0]),
    )
    plt.imshow(nuc_seg_arr.squeeze()[test_slice], cmap="gray")
    num_nuclei_in_crop = np.unique(nuc_seg_arr.squeeze()[test_slice]).size

    break


def get_num_nuclei_in_crop(
    nuclei_coords_Y: tuple[np.ndarray, np.ndarray],
    nuclei_coords_X: tuple[np.ndarray, np.ndarray],
    bounds_Y: tuple[np.ndarray],
    bounds_X: tuple[np.ndarray],
) -> np.ndarray:
    """
    whole_mask_indices_Y and centroid_indices_Y have the shape:
    (n_unique_labels x n_coords_per_label)
    df["start_y"] has the shape (n_crops,)
    """

    coord_in_Y_bounds = np.logical_and(
        np.reshape(whole_mask_indices_Y, (*whole_mask_indices_Y_arr.shape, 1))
        >= np.reshape(df["start_y"], (1, 1, len(df["start_y"]))),
        np.reshape(whole_mask_indices_Y, (*whole_mask_indices_X_arr.shape, 1))
        <= np.reshape(df["end_y"], (1, 1, len(df["end_y"]))),
    )
    coord_in_Y_bounds = coord_in_Y_bounds.any(axis=1)

    coord_in_X_bounds = np.logical_and(
        np.reshape(whole_mask_indices_X, (*whole_mask_indices_Y_arr.shape, 1))
        >= np.reshape(df["start_x"], (1, 1, len(df["start_x"]))),
        np.reshape(whole_mask_indices_X, (*whole_mask_indices_X_arr.shape, 1))
        <= np.reshape(df["end_x"], (1, 1, len(df["end_x"]))),
    )
    coord_in_X_bounds = coord_in_X_bounds.any(axis=1)

    num_nuclei_in_crop = np.logical_and(coord_in_Y_bounds, coord_in_X_bounds).sum(axis=0)

    # for getting number of nuclei from arbitrary array
    test_slice = (
        slice(df["start_y"].iloc[0], df["end_y"].iloc[0]),
        slice(df["start_x"].iloc[0], df["end_x"].iloc[0]),
    )
    plt.imshow(nuc_seg_arr.squeeze()[test_slice], cmap="gray")
    num_nuclei_in_crop = np.unique(nuc_seg_arr.squeeze()[test_slice]).size

    return num_nuclei_in_crop


# def get_num_nuclei_in_
