import fire
import numpy as np
import pandas as pd

from bioio.writers import OmeTiffWriter
from pathlib import Path

from cellsmap.util.manifest_io import load_pca_model
from cellsmap.model_features.apply_model import apply_model
from cellsmap.util.set_output import get_output_path


def generate_spatial_pc_movie(model_name:str, dataset_name: str, pca_dir:str, overlap: float = 0.75, resolution_level:int=0):
    """
    Function to generate a spatial movie of PCA features from a model's predictions. Saves out a `timepoint * pc  * y * x` tiff file for each position in the dataset.
    The movie is saved in the `models/{model_name}/spatial_pcs` directory.

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
    """
    # apply model with specified overlap
    overrides = {
        "model.spatial_inferer.splitter.overlap": overlap
    }
    feats_path = apply_model(model_name, dataset_name, resolution_level=resolution_level, overrides=overrides, upload_to_fms=False)
    
    # load model predictions and apply PCA
    data = pd.read_parquet(feats_path)
    feat_cols = [c for c in data.columns if c.startswith('feat_')]
    feat_cols = sorted(feat_cols, key=lambda x: int(x.split('_')[1]))

    pca = load_pca_model(pca_dir)
    pca_feats = pca.transform(data[feat_cols].values)
    pc_columns = [f'pc{i}' for i in range(pca_feats.shape[1])]
    data[pc_columns] = pca_feats

    # how much crop moves in each direction during sliding window inference is the second smallest value of start_x and start_y (first value is 0)
    step_x = sorted(data.start_x.unique())[1]
    step_y = sorted(data.start_y.unique())[1]

    # convert start_x and start_y to indices
    data.start_x = data.start_x // step_x
    data.start_y = data.start_y // step_y

    n_timepoints = data.frame_number.max() + 1
    n_pcs = pca_feats.shape[1]

    save_dir = Path(get_output_path(f'models/{model_name}/spatial_pcs'))
    for position_name, position_data in data.groupby('position'):
        # fill in movie with pc values in location of (start_y, start_x)
        movie = np.zeros((n_timepoints, n_pcs, data.start_y.max()+1, data.start_x.max()+1))
        for T in range(n_timepoints):
            sub = position_data[(position_data.frame_number == T)]
            movie[T, :, sub.start_y, sub.start_x] = sub[pc_columns].values
        OmeTiffWriter.save(uri = save_dir/f'{dataset_name}_{position_name}.tiff', data = movie)

if __name__ == '__main__':
    fire.Fire(generate_spatial_pc_movie)