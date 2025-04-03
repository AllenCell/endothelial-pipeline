from cellsmap.util.dataset_io import get_model_info
import torch
from typing import List
from pathlib import Path
from cellsmap.model_features.utils.mlflow_utils import load_mlflow_model
from cellsmap.util.set_ouput import get_output_path


def generate_from_coords(model_name, coords:List[List[float]], n_noise_samples=1, average=False):
    mlflow_id = get_model_info(model_name)['mlflow_run_id']
    model_path = get_output_path('models')
    model = load_mlflow_model(mlflow_id, save_path=model_path)

    coords  = torch.from_numpy(coords).float()

    if torch.cuda.is_available():
        coords = coords.to("cuda")
        model = model.to("cuda")
    walk_img = model.generate_from_latent(coords, n_noise_samples=n_noise_samples, average=average, save=False)
    return walk_img