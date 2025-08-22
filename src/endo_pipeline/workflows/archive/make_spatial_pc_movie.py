from typing import Any

import dask
import pandas as pd

FLUOR_CHANNEL = 0
BF_CHANNEL = 1


def make_overlay(filename, feature_movie, end_y, end_x):
    """Make an overlay of the feature movie with the brightfield and fluorescent channels."""
    import dask.array as da
    import numpy as np
    from bioio import BioImage

    img = BioImage(filename)
    img.set_resolution_level(1)
    n_t = range(feature_movie.shape[0])
    fluor_img = img.get_image_dask_data("TZYX", C=FLUOR_CHANNEL, T=n_t).max(1).astype(np.float32)
    bf_img = img.get_image_dask_data("TZYX", C=BF_CHANNEL, T=n_t).std(1).astype(np.float32)

    # crop movie to only include data used for feature extraction
    fluor_img = fluor_img[:, :end_y, :end_x][:, None]
    bf_img = bf_img[:, :end_y, :end_x][:, None]
    feature_movie = da.concatenate((fluor_img, bf_img, feature_movie), axis=1)
    # for ometiff saving, add dummy Z dimension
    feature_movie = da.expand_dims(feature_movie, 2)
    return feature_movie


@dask.delayed
def create_frame(shape, df, feat_cols):
    """Create a frame of spatial features from a DataFrame."""
    import numpy as np

    timepoint_movie = np.zeros(shape)
    count_movie = np.zeros(shape)
    coords = df[["start_y", "end_y", "start_x", "end_x"]].values
    values = df[feat_cols].values[:, :, None, None]
    for i in range(values.shape[0]):
        # fill in movie with pc values in crop location
        timepoint_movie[:, coords[i, 0] : coords[i, 1], coords[i, 2] : coords[i, 3]] += values[i]
        count_movie[:, coords[i, 0] : coords[i, 1], coords[i, 2] : coords[i, 3]] += 1
    return timepoint_movie / count_movie


def get_physical_pixel_sizes(filename):
    """Get resolution level 1 physical pixel sizes from a zarr file."""
    from bioio import BioImage

    im = BioImage(filename)
    im.set_resolution_level(1)
    return im.physical_pixel_sizes


def _get_per_cell_features(
    data: pd.DataFrame,
    feat_cols: list[str],
    dataset_name: str,
    position: str,
    timepoint: str,
) -> pd.DataFrame:
    """
    Rearrange a list of spatial features into an image and
    extract the mean value within each segmentation region.
    """
    import numpy as np
    from bioio import BioImage
    from skimage.measure import regionprops_table

    from src.endo_pipeline.manifests import get_image_location_for_dataset, load_image_manifest

    movie_shape_y, movie_shape_x = data.end_y.max(), data.end_x.max()

    spatial_pcs = create_frame(
        (len(feat_cols), movie_shape_y, movie_shape_x), data, feat_cols
    ).compute()

    manifest = load_image_manifest("cdh5_classic_seg")
    segmentation_path = get_image_location_for_dataset(
        manifest, dataset_name, int(position[1:]), int(timepoint) // 6
    )

    # TODO set resolution to 1 once segmentations are zarrs
    segmentation = BioImage(segmentation_path).get_image_dask_data("YX").compute()
    segmentation = segmentation[::2, ::2]

    # pad spatial_pcs to segmentation size with nans
    # so cells overlapping the edge of the image have nan for mean
    spatial_pcs = np.pad(
        spatial_pcs,
        (
            (0, 0),
            (0, segmentation.shape[-2] - spatial_pcs.shape[1]),
            (0, segmentation.shape[-1] - spatial_pcs.shape[2]),
        ),
        mode="constant",
        constant_values=np.nan,
    )
    # get the regionprops of the segmentation
    measurement_cols = ["intensity_mean"]
    props = pd.DataFrame(
        regionprops_table(
            segmentation,
            intensity_image=spatial_pcs.transpose(1, 2, 0),
            properties=["label", measurement_cols],
        )
    )

    rename_dict = {}
    for measurement_name in measurement_cols:
        for i, col in enumerate(feat_cols):
            rename_dict[f"{measurement_name}-{i}"] = f"{measurement_name}_{col}"
    props.rename(columns=rename_dict, inplace=True)
    props["position"] = position
    props["frame_number"] = timepoint
    props["dataset"] = dataset_name
    return props


def get_feats(
    model_name: str,
    dataset_name: str,
    save_dir: str,
    overlap: float = 0.75,
    resolution_level: int = 0,
    overrides: dict[str, Any] | None = None,
    use_pca: bool = False,
    n_pcs: int = 8,
):
    """
    Apply a model to a dataset and return the features.

    If PCA is used, apply PCA to the features.
    """
    from src.endo_pipeline.library.analyze.diffae_manifest import (
        fit_pca,
        get_feature_column_names,
        get_pc_column_names,
        project_manifest_to_pcs,
    )
    from src.endo_pipeline.library.model import (
        apply_model_on_grid_of_crops_from_one_dataset,
        load_overrides,
    )

    overrides = load_overrides(overrides)
    # apply model with specified overlap
    overrides.update({"model.spatial_inferer.splitter.overlap": overlap})
    feats_path = apply_model_on_grid_of_crops_from_one_dataset(
        model_name,
        dataset_name,
        resolution_level=resolution_level,
        overrides=overrides,
        save_path=save_dir,
        upload_to_fms=False,
    )
    # load model predictions and apply PCA
    data = pd.read_parquet(feats_path)
    feat_cols = get_feature_column_names(data)

    if use_pca:
        # if PCA is specified, fit PCA to the model features
        pca = fit_pca(model_name=model_name, num_pcs=n_pcs)
        # add PC component features to the dataframe
        data = project_manifest_to_pcs(data, pca, feat_cols=feat_cols)
        # use the PCA components as the features
        feat_cols = get_pc_column_names(data, pc_axes=range(n_pcs or pca.n_components_))

    return data, feat_cols


def generate_spatial_feature_movie(
    model_name: str,
    dataset_name: str,
    use_pca: bool = False,
    overlap: float = 0.75,
    resolution_level: int = 1,
    n_pcs: int = 8,
    overrides: dict[str, Any] | None = None,
):
    """
    Generate a spatial movie of PCA features from a model's predictions.

    The movie is saved in the `models/{model_name}/spatial_pcs/{dataset_name}` directory.
    """
    import dask.array as da
    import numpy as np

    from src.endo_pipeline.io import get_output_path
    from src.endo_pipeline.library.process.convert_to_zarr.write_zarr import write_scene

    save_dir = get_output_path(
        "models", model_name, "spatial_pcs", dataset_name, include_timestamp=False
    )

    data, feat_cols = get_feats(
        model_name,
        dataset_name,
        save_dir,
        overlap=overlap,
        resolution_level=resolution_level,
        overrides=overrides,
        use_pca=use_pca,
        n_pcs=n_pcs,
    )

    n_features = len(feat_cols)
    movie_shape_y, movie_shape_x = data.end_y.max(), data.end_x.max()
    n_timepoints = data.frame_number.max() + 1

    physical_pixel_sizes = get_physical_pixel_sizes(data.zarr_path.iloc[0])

    for position_name, position_data in data.groupby("position"):
        frame_shape = (n_features, movie_shape_y, movie_shape_x)
        movie = da.stack(
            [
                da.from_delayed(
                    create_frame(
                        frame_shape,
                        position_data[position_data.frame_number == T],
                        feat_cols,
                    ),
                    shape=frame_shape,
                    dtype=np.float32,
                )
                for T in range(n_timepoints)
            ]
        )
        movie = make_overlay(
            position_data.zarr_path.iloc[0],
            movie,
            end_y=data.end_y.max(),
            end_x=data.end_x.max(),
        )
        write_scene(
            movie,
            channels=["Fluor", "BF", *feat_cols],
            full_zarr_path=str(save_dir / f"{dataset_name}_{position_name}.zarr"),
            dataset=dataset_name,
            position=position_name,
            # half resolution
            physical_pixel_sizes=physical_pixel_sizes,
            interval_min=5.0,
            # don't create multi-resolution zarr
            xy_scaling=[],
            z_scaling=[],
        )


def measure_per_cell_features(
    dataset_name: str,
    model_name: str,
    overlap: float = 0.9,
    resolution_level: int = 1,
    upload_to_fms: bool = False,
    n_pcs: int = 8,
    use_pca: bool = False,
):
    """Take within-mask mean of each feature for each cell in the segmentation."""
    from src.endo_pipeline.configs import load_dataset_config, load_model_config
    from src.endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from src.endo_pipeline.manifests import (
        DataframeLocation,
        DataframeManifest,
        load_dataframe_manifest,
        save_dataframe_manifest,
    )

    save_dir = get_output_path(
        "models", model_name, "cell_pcs", dataset_name, include_timestamp=False
    )
    data, feat_cols = get_feats(
        model_name,
        dataset_name,
        save_dir,
        overlap=overlap,
        resolution_level=resolution_level,
        use_pca=use_pca,
        n_pcs=n_pcs,
    )
    spatial_pc_info = []

    for position_name, position_data in data.groupby("position"):
        for t, timepoint_data in position_data.groupby("frame_number"):
            spatial_pc_info.append(
                _get_per_cell_features(timepoint_data, feat_cols, dataset_name, position_name, t)
            )
    spatial_pc_info = pd.concat(spatial_pc_info)

    feats_path = save_dir / f"{dataset_name}_cell_features.parquet"
    spatial_pc_info.to_parquet(feats_path)

    if upload_to_fms:
        model_config = load_model_config(model_name)
        dataset_config = load_dataset_config(dataset_name)

        dataset_annotations = build_fms_annotations(
            dataset_config,
            model=model_config,
        )

        # upload prediction file to FMS and get file ID
        file_id = upload_file_to_fms(
            feats_path,
            annotations=dataset_annotations,
            file_type="parquet",
        )

        # Store FMS ID in dataframe manifest

        manifest_name = "cell_mean_features"
        workflow_name = "measure_per_cell_features"

        try:
            manifest = load_dataframe_manifest(manifest_name)
        except FileNotFoundError:
            manifest = DataframeManifest(name=manifest_name, workflow=workflow_name)

        manifest.locations[dataset_config.name] = DataframeLocation(fmsid=file_id)
        save_dataframe_manifest(manifest)


def main(
    model_name: str,
    dataset_name: str,
    resolution_level: int = 1,
    n_pcs: int = 8,
    use_pca: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    """
    Make a spatial feature movie and measure per-cell features for visualization on TFE.

    Parameters
    ----------
    model_name
        Name of the model from to apply.
    dataset_name
        Name of the dataset from to apply the model to.
    use_pca
        Project the model features to PCA components.
    overlap
        Overlap between sliding windows during inference.
        Higher overlaps will givemore spatial resolution but take longer for inference.
    resolution_level
        Resolution level to apply the model at.
    n_pcs
        Number of PCA components to use.
        This argument is only used if `use_pca` is not None.
    overrides
        Dictionary of overrides to apply to the model.

    Returns
    -------
    :
        Saves out a zarr file for each position in the dataset with an overlay of
        the features on the brightfield standard deviation projection and max projection
        of the fluorescent channel. Also saves out a parquet file with per-cell features
        for each timepoint in the dataset and uploads it to FMS.

    """
    generate_spatial_feature_movie(
        model_name=model_name,
        dataset_name=dataset_name,
        resolution_level=resolution_level,
        n_pcs=n_pcs,
        use_pca=use_pca,
        overrides=overrides,
    )

    measure_per_cell_features(
        model_name=model_name,
        dataset_name=dataset_name,
        resolution_level=resolution_level,
        n_pcs=n_pcs,
        use_pca=use_pca,
        upload_to_fms=True,
    )


if __name__ == "__main__":
    from src.endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
