import concurrent
import re
from pathlib import Path
from typing import Any, Literal, Sequence

import dask.array as dd
import numpy as np
import pandas as pd

# for data exploration; remove later
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from skimage.measure import regionprops
from tqdm import tqdm

from src.endo_pipeline.configs import load_dataset_collection_config
from src.endo_pipeline.configs.dataset_io import load_nuclei_prediction
from src.endo_pipeline.io import get_output_path, save_plot_to_path
from src.endo_pipeline.library.analyze.diffae_manifest.diffae_manifest_utils import get_valid_subset
from src.endo_pipeline.library.analyze.integration.track_integration import (  # get_approx_point_from_grid,; get_approx_vec_from_grid,; get_gridcrop_and_cellcentric_trajectories_and_flow_fields,; get_vector_angles_as_grid,; get_vector_dot_products_as_grid,; get_vector_vector_angle_fast,; make_angular_deviation_test,
    get_preprocessed_manifests_and_km_bounds,
)
from src.endo_pipeline.library.process.general_image_preprocessing import (
    get_default_dim_order,
    sequence_to_scalar,
)


def adjust_crop_bounds_to_0th_bin_level(
    merged_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adjust the crop bounds to the 0th level of resolution for the imaging data.
    """
    # adjust the crop bounds to the 0th level of resolution
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
    return merged_feats_df


def get_nuclei_coords(
    props: regionprops,  # type:ignore
    props_dim_order: str,
    kind: Literal["centroid", "all"] = "centroid",
) -> dict[str, np.ndarray]:
    if kind == "all":
        # find the largest nuclei in the image because we
        # will need it for padding the coordinates later
        biggest_nuc_mask = max([p.num_pixels for p in props])  # type:ignore

    # centroids
    nuclei_coords: dict = {f"coords_{d}": [] for d in props_dim_order}
    for p in props:  # type:ignore
        match kind:
            case "centroid":
                # get only nuclei centroids
                # the ndmin=2 is so that the p.centroid shape is the same as p.coords
                # and will work in the function `get_num_nuclei_in_crops` correctly
                coords = np.array(p.centroid, ndmin=2).astype(float)
            case "all":
                # get all the nuclei coordinates
                coords = p.coords.astype(float)
                # define how much padding you need to add to these nuclei coordinates
                pad_width = ((0, biggest_nuc_mask - p.coords.shape[0]), (0, 0))
                # do the padding
                coords = np.pad(coords, pad_width, mode="constant", constant_values=np.nan)
        for dim in props_dim_order:
            nuclei_coords[f"coords_{dim}"].append(coords[..., props_dim_order.index(dim)])
    nuclei_coords_arrs = {dim: np.stack(coords).squeeze() for dim, coords in nuclei_coords.items()}

    return nuclei_coords_arrs


def get_num_unique_values_in_bounds_from_df(
    nuclei_coords_Y: pd.Series,
    nuclei_coords_X: pd.Series,
    crop_bounds_Y: tuple[pd.Series, pd.Series],
    crop_bounds_X: tuple[pd.Series, pd.Series],
) -> np.ndarray:
    """
    nuclei_coords_Y and nuclei_coords_X have the shape:
    (n_crops x n_unique_labels)
    crop_bounds_Y has the shape (n_crops, 2)
    """
    start_y, end_y = crop_bounds_Y
    start_x, end_x = crop_bounds_X

    coord_in_Y_bounds = np.logical_and(
        nuclei_coords_Y >= np.reshape(start_y, (len(start_y), 1)),  # type:ignore[arg-type]
        nuclei_coords_Y <= np.reshape(end_y, (len(end_y), 1)),  # type:ignore[arg-type]
    )
    coord_in_X_bounds = np.logical_and(
        nuclei_coords_X >= np.reshape(start_x, (len(start_x), 1)),  # type:ignore[arg-type]
        nuclei_coords_X <= np.reshape(end_x, (len(end_x), 1)),  # type:ignore[arg-type]
    )
    num_nuclei_in_crop = np.logical_and(coord_in_Y_bounds, coord_in_X_bounds).sum(axis=1)

    return num_nuclei_in_crop


# unused: get_num_unique_values_in_bounds
def get_num_unique_values_in_bounds(
    nuclei_coords_Y: np.ndarray,
    nuclei_coords_X: np.ndarray,
    crop_bounds_Y: tuple[np.typing.ArrayLike, np.typing.ArrayLike],
    crop_bounds_X: tuple[np.typing.ArrayLike, np.typing.ArrayLike],
) -> np.ndarray:
    """
    nuclei_coords_Y and nuclei_coords_X have the shape:
    (n_unique_labels x n_coords_per_label)
    crop_bounds_Y has the shape (n_crops, 2)
    """
    start_y, end_y = crop_bounds_Y
    start_x, end_x = crop_bounds_X

    coord_in_Y_bounds = np.logical_and(
        np.reshape(nuclei_coords_Y, (*nuclei_coords_Y.shape, 1))
        >= np.reshape(start_y, (1, 1, len(start_y))),  # type:ignore[arg-type]
        np.reshape(nuclei_coords_Y, (*nuclei_coords_Y.shape, 1))
        <= np.reshape(end_y, (1, 1, len(end_y))),  # type:ignore[arg-type]
    )
    coord_in_Y_bounds = coord_in_Y_bounds.any(axis=1)

    coord_in_X_bounds = np.logical_and(
        np.reshape(nuclei_coords_X, (*nuclei_coords_X.shape, 1))
        >= np.reshape(start_x, (1, 1, len(start_x))),  # type:ignore[arg-type]
        np.reshape(nuclei_coords_X, (*nuclei_coords_X.shape, 1))
        <= np.reshape(end_x, (1, 1, len(end_x))),  # type:ignore[arg-type]
    )
    coord_in_X_bounds = coord_in_X_bounds.any(axis=1)

    num_nuclei_in_crop = np.logical_and(coord_in_Y_bounds, coord_in_X_bounds).sum(axis=0)

    return num_nuclei_in_crop


# unused: get_num_nuclei_in_array
def get_num_nuclei_in_array(img_arr: np.ndarray | dd.Array, crop: tuple[slice, ...] | None) -> int:
    """
    Get the number of labeled nuclei in an array or dask array.
    Array will be cropped before counting nuclei if crop is provided.
    If there is even 1 pixel of a labeled nuclei then it will be counted,
    therefore you may want to create an image of the centroids or cleaned
    up nuclei before counting.
    """
    if crop is not None:
        img_arr = img_arr[crop]

    if isinstance(img_arr, dd.Array):
        num_nuclei = np.unique(img_arr.compute()).size
        return num_nuclei
    elif isinstance(img_arr, np.ndarray):
        num_nuclei = np.unique(img_arr).size
        return num_nuclei
    else:
        raise TypeError(f"Unsupported type: {type(img_arr)}")


def compute_nuclei_centroids(
    dataset_name: str,
    position: int,
    timeframe: int,
) -> dict:

    # get the nuclei prediction
    dim_order = get_default_dim_order()
    nuc_seg = load_nuclei_prediction(
        dataset_name=dataset_name,
        position=position,
        T=timeframe,
        dim_order=dim_order,
    )

    # get nuclei segmentation properties and dimension order of those properties
    nuc_seg_arr = nuc_seg.compute()
    props = regionprops(nuc_seg_arr.squeeze())
    dim_shapes = dict(zip(dim_order, nuc_seg.shape))
    dim_order_squeezed = "".join([d for d in dim_order if dim_shapes[d] > 1])

    # centroid_indices_Y_arr, centroid_indices_X_arr = get_nuclei_coords(
    centroids: dict[str, Any] = get_nuclei_coords(
        props=props,
        props_dim_order=dim_order_squeezed,
        kind="centroid",
    )
    centroids["dataset_name"] = dataset_name
    centroids["position"] = position
    centroids["image_index"] = timeframe

    return centroids


def compute_nuclei_centroids_multiproc(args: tuple[str, int, int]) -> dict:
    return compute_nuclei_centroids(*args)


def add_num_nuclei_in_crop_column(
    merged_feats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the number of nuclei in each crop to the merged features DataFrame.
    """

    groups = merged_feats_df.groupby(["dataset_name", "position", "image_index"])

    # get the nuclei coordinates
    nuclei_centroids_dir = get_output_path(
        Path(__file__).stem, "nuclei_coords", include_timestamp=False
    )
    dataset_name = sequence_to_scalar(merged_feats_df["dataset_name"])
    nuclei_centroids_path = nuclei_centroids_dir / f"{dataset_name}_nuclei_centroids.parquet"

    # if the nuclei coordinates are already computed, load them
    if nuclei_centroids_path.exists():
        nuc_centroid_indices = pd.read_parquet(nuclei_centroids_path)
    # otherwise, compute and save them
    # (this will take about 60 minutes divided by n_cores used)
    else:
        # compute the nuclei prediction centroids
        max_cores = None
        args = groups.groups.keys()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            results = list(
                tqdm(
                    executor.map(compute_nuclei_centroids_multiproc, args),
                    total=len(groups),
                    desc="Computing nuclei centroids",
                )
            )
        # convert results to DataFrame
        nuc_centroid_indices = pd.DataFrame(results)
        # save results for so this step doesn't have to be rerun each time
        nuc_centroid_indices.to_parquet(nuclei_centroids_path, index=False)

    # combine the nuclei centroids with the merged features DataFrames
    merged_feats_df = pd.merge(
        merged_feats_df,
        nuc_centroid_indices,
        on=["dataset_name", "position", "image_index"],
        how="left",
    )
    groups = merged_feats_df.groupby(["dataset_name", "position", "image_index"])

    num_nuclei_in_crop = []
    for nm, df in tqdm(groups, desc="Counting nuclei in crops"):
        # get the number of nuclei in the crops at each timepoint
        num_nuc_centroids = get_num_unique_values_in_bounds_from_df(
            nuclei_coords_Y=np.stack(list(df["coords_Y"])),
            nuclei_coords_X=np.stack(list(df["coords_X"])),
            crop_bounds_Y=(df["start_y"], df["end_y"]),
            crop_bounds_X=(df["start_x"], df["end_x"]),
        )
        num_nuclei_in_crop.append(pd.Series(num_nuc_centroids, index=df.index))

    merged_feats_df["num_nuclei_in_crop"] = pd.concat(
        num_nuclei_in_crop, axis=0, ignore_index=False
    )
    return merged_feats_df


def get_correlation_matrix_df(
    features_df: pd.DataFrame,
    column_names_for_x_axis: list[str],
    column_names_for_y_axis: list[str],
    name_of_x_axis: str,
    name_of_y_axis: str,
    df_format: Literal["long", "wide-corrcoeff", "wide-pval"] = "long",
) -> pd.DataFrame:
    """
    Get the Pearson correlations between each column in `column_names_for_x_axis`
    compared with each column in `column_names_for_y_axis`.
    This is used to compare the diffae features and the measured features,
    and then used again to compare the PCs and the measured features.
    If `df_format` is one of the "wide" options then the outputted dataframe
    of this function can be passed directly to `seaborn.heatmap` or
    `seaborn.clustermap` for visualization.

    Parameters
    ----------
    features_df : pd.DataFrame
        The DataFrame containing the features to correlate.
    column_names_for_x_axis : list[str]
        The names of the columns to use for the x-axis.
    column_names_for_y_axis : list[str]
        The names of the columns to use for the y-axis.
    name_of_x_axis : str
        The name of the x-axis.
    name_of_y_axis : str
        The name of the y-axis.
    df_format : Literal["long", "wide"], optional
        The format of the output DataFrame. If "long", the output DataFrame will have columns:
        - name_of_y_axis
        - name_of_x_axis
        - pearsonr
        - pval
        If "wide-corrcoeff", the output DataFrame will have a column for each column in
        column_names_for_x_axis and the index will be the column names in
        column_names_for_y_axis, with the values in the DataFrame corresponding to the
        correlation coefficients from the "long" version of the table.
        "wide-pval" is similar to "wide-corrcoeff" but the values correspond to the p-values.
        Defaults to "long".
    """
    records = []
    for col_for_y in column_names_for_y_axis:
        for col_for_x in column_names_for_x_axis:
            valid_records = np.isfinite(features_df[[col_for_y, col_for_x]]).all(axis=1)
            corr, pval = pearsonr(
                features_df[col_for_y][valid_records],
                features_df[col_for_x][valid_records],
            )
            records.append(
                {
                    name_of_y_axis: col_for_y,
                    name_of_x_axis: col_for_x,
                    "pearsonr": corr,
                    "pval": pval,
                }
            )
    correlation_df = pd.DataFrame(records)

    if df_format in ("wide-corrcoeff", "wide-pval"):
        if df_format == "wide-corrcoeff":
            value_col = "pearsonr"
        elif df_format == "wide-pval":
            value_col = "pval"
        correlation_df = correlation_df.pivot(
            index=name_of_y_axis,
            columns=name_of_x_axis,
            values=value_col,
        )
        correlation_df = correlation_df[column_names_for_x_axis]  # sort the columns
        correlation_df = correlation_df.reindex(index=column_names_for_y_axis)  # sort the index
    elif df_format == "long":
        pass
    else:
        raise ValueError(
            f"Unsupported df_format: {df_format}. Supported formats are 'long', 'wide-corrcoeff', 'wide-pval'."
        )
    return correlation_df


if __name__ == "__main__":
    dataset_name_list = load_dataset_collection_config("pca_reference").datasets
    # dataset_name = dataset_name_list[0]  # for testing purposes
    for dataset_name in dataset_name_list:
        # out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

        # load and preprocess the different diffae manifests and PCA pipeline
        # NOTE: this takes a little over a minute to load
        merged_feats_df, diffae_grid_crops, bounds = get_preprocessed_manifests_and_km_bounds(
            dataset_name, datasets_for_bounds=dataset_name_list
        )

        # ensure the crop coordinates have an integer datatype
        merged_feats_df = adjust_crop_bounds_to_0th_bin_level(merged_feats_df)

        # add the number of nuclei columns
        merged_feats_df = add_num_nuclei_in_crop_column(merged_feats_df)

        out_subdir = get_output_path(Path(__file__).stem, dataset_name, include_timestamp=False)

        # get the names of the columns that you are interested in using
        pc_col_names = []
        diffae_feature_col_names = []
        for col_nm in merged_feats_df.columns:
            if re.match("pc[0-9]", col_nm):
                pc_col_names.append(col_nm)
            elif re.match("feat_[0-9]+", col_nm):
                diffae_feature_col_names.append(col_nm)
            else:
                continue

        measured_col_names = [
            "alignment_deg_rel_to_flow",
            "nematic_order",
            "area",
            "perimeter",
            "eccentricity",
            "aspect_ratio",
            "cell_fluorescence_mean (a.u.)",
            "num_nuclei_in_crop",
            "cell_solidity",
            "number_of_neighbors",
            "nuc_pos_rel_cell_angle_deg",
        ]
        assert all(np.isin(measured_col_names, merged_feats_df.columns)), (
            f"Not all measured_col_names are in merged_feats_df. "
            f"Missing: {set(measured_col_names) - set(merged_feats_df.columns)}"
        )

        for meas in measured_col_names:
            if not np.isfinite(merged_feats_df[meas]).all():
                print(f"{meas} contains non-finite values")

        # 1. find which of the diffae features correlate with which measured features
        correlation_df_feats = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=diffae_feature_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="feature",
            df_format="wide-corrcoeff",
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_feats, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_feats_vs_measured_feats",
        )

        correlation_df_pcs = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_pcs, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_pcs_vs_measured_feats",
        )

        # repeat the above correlations but filter data table
        # to only include the steady state timepoints
        merged_feats_df = get_valid_subset(
            merged_feats_df,
            dataset_name=dataset_name,
            verbose=False,
        )

        correlation_df_feats = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=diffae_feature_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="feature",
            df_format="wide-corrcoeff",
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_feats, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_feats_vs_measured_feats_steady_state",
        )

        correlation_df_pcs = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="measurement",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df_pcs, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
        save_plot_to_path(
            figure=fig,
            output_path=out_subdir,
            figure_name=f"{dataset_name}_correlation_pcs_vs_measured_feats_steady_state",
        )

        # 2. plot the PC loadings
        # from src.endo_pipeline.workflows.development.visualize_pca_attributes import (
        #     plot_component_loadings,
        # )

        # 3. correlations between diffae features and PCs
        # (NOTE THAT GETTING THE PC LOADINGS IS THE CORRECT APPROACH)
        correlation_df = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=diffae_feature_col_names,
            column_names_for_y_axis=pc_col_names,
            name_of_x_axis="feature",
            name_of_y_axis="PC",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)

        # 3. correlate the measured features with each other
        # to see if any measures tend to co-occur
        correlation_df = get_correlation_matrix_df(
            features_df=merged_feats_df,
            column_names_for_x_axis=measured_col_names,
            column_names_for_y_axis=measured_col_names,
            name_of_x_axis="measure1",
            name_of_y_axis="measure2",
            df_format="wide-corrcoeff",
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlation_df, annot=True, cmap="RdBu", center=0, vmin=-1, vmax=1)
