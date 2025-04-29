import fire
import numpy as np
import pandas as pd
import dask.array as da

from typing import Dict, Any, Optional
from bioio.writers import OmeTiffWriter
from bioio import BioImage
from pathlib import Path
from tqdm import trange

from cellsmap.util.manifest_io import load_pca_model
from cellsmap.model_features.apply_model import apply_model, load_overrides
from cellsmap.util.set_output import get_output_path

FLUOR_CHANNEL = 0
BF_CHANNEL = 1

def make_overlay(filename, feature_movie, end_y, end_x):
    img =BioImage(filename)
    img.set_resolution_level(1)
    n_t = range(feature_movie.shape[0])
    fluor_img = img.get_image_dask_data('TZYX', C=FLUOR_CHANNEL, T = n_t).max(1)
    bf_img = img.get_image_dask_data('TZYX', C=BF_CHANNEL, T = n_t).std(1)

    # crop movie to only include data used for feature extraction
    fluor_img = fluor_img[:, :end_y, :end_x]
    bf_img = bf_img[:, :end_y, :end_x]

    feature_movie = da.concatenate((fluor_img[:, None], bf_img[:, None], feature_movie), axis=1)
    # for ometiff saving, add dummy Z dimension
    feature_movie = da.expand_dims(feature_movie, 2)
    return feature_movie


def generate_spatial_feature_movie(model_name:str, dataset_name: str, pca_dir:str, overlap: float = 0.75, resolution_level:int=0, n_pcs: Optional[int] = None, use_pcs: bool = True, overrides: Dict[str, Any] = {}):
    """
    Function to generate a spatial movie of PCA features from a model's predictions. Saves out a `timepoint * pc  * y * x` tiff file for each position in the dataset with an overlay of the brightfield standard deviation projection and max projection of the fluorescent channel.
    The movie is saved in the `models/{model_name}/spatial_pcs` directory.

    Example usage: python make_spatial_pc_movie.py --model_name diffae_04_10 --dataset_name 20241016_20X --pca_dir //allen/aics/users/erin.angelini/git-repos/cellsmap/results/stochastic_dynamics/default/outputs/ --overlap 0.5 --resolution_level 0 --n_pcs 3

    Parameters
    ----------
    model_name: str
        Name of the model from `model_config.yaml` to apply.
    dataset_name: str
        Name of the dataset from `data_config.yaml` to apply the model to.
    pca_dir: str
        Directory where a fitted PCA model is stored.
    overlap: float
        Overlap between sliding windows during inference. Default is 0.75. Higher overlaps will givemore spatial resolution but take longer for inference.
    resolution_level: int
        Resolution level to apply the model at. Default is 0 (highest resolution)
    n_pcs: Optional[int]
        Number of PCA components to use. Default is None, which will use all components.This argument is only used if `use_pcs` is True.
    use_pcs: bool
        Whether to use PCA components. If False, will use the original features. Default is True.
    overrides: Dict[str, Any]
        Dictionary of overrides to apply to the model. Default is {}.
    """
    # save_dir = Path(get_output_path(f'models/{model_name}/spatial_pcs'))
    save_dir = Path(get_output_path(f'models/{model_name}/benji_test'))
    overrides = load_overrides(overrides)
    # apply model with specified overlap
    overrides.update({
        "model.spatial_inferer.splitter.overlap": overlap
    })
    feats_path = apply_model(model_name, dataset_name, resolution_level=resolution_level, overrides=overrides, save_path=save_dir, upload_to_fms=False)
    # load model predictions and apply PCA
    data = pd.read_parquet(feats_path)
    feat_cols = [c for c in data.columns if c.startswith('feat_')]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split('_')[1]))

    if use_pcs: 
        pca = load_pca_model(pca_dir)
        feats = pca.transform(data[feat_cols].values)

        n_pcs = n_pcs or feats.shape[1]
        if n_pcs > feats.shape[1]:
            raise ValueError(f"n_pcs {n_pcs} is greater than the number of PCA components {feats.shape[1]}")
        
        pc_columns = [f'pc{i}' for i in range(n_pcs)]
        data[pc_columns] = feats[:, :n_pcs]
        # use the PCA components as the features
        feat_cols = pc_columns

    movie_shape_y, movie_shape_x = data.end_y.max(), data.end_x.max()
    n_timepoints = data.frame_number.max() + 1

    for position_name, position_data in data.groupby('position'):
        movie = np.zeros((n_timepoints, n_pcs, movie_shape_y, movie_shape_x))
        for T in trange(n_timepoints):
            timepoint_movie = np.zeros((n_pcs, movie_shape_y, movie_shape_x))
            count_movie= np.zeros((n_pcs, movie_shape_y, movie_shape_x))

            sub = position_data[(position_data.frame_number == T)]
            for row in sub.itertuples():
                # fill in movie with pc values in crop location
                timepoint_movie[:, row.start_y:row.end_y, row.start_x:row.end_x] += np.array([getattr(row, col) for col in feat_cols])[:, None, None]
                count_movie[:, row.start_y:row.end_y, row.start_x:row.end_x] += 1
            movie[T] = timepoint_movie / count_movie
        movie = make_overlay(data.zarr_path.iloc[0], movie, end_y=data.end_y.max(), end_x=data.end_x.max())
        breakpoint()
        OmeTiffWriter.save(uri = save_dir/f'{dataset_name}_{position_name}.tiff', data = movie)

if __name__ == '__main__':
    # fire.Fire(generate_spatial_pc_movie)
    model_name = "diffae_04_10"
    pca_dir = "//allen/aics/users/erin.angelini/git-repos/cellsmap/results/stochastic_dynamics/default/outputs/"
    overlap = 0.5
    for dataset_name in ["20241016_20X"]:
        generate_spatial_feature_movie(model_name=model_name, dataset_name=dataset_name, pca_dir=pca_dir, overlap=overlap, n_pcs=3)
