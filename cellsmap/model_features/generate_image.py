import mlflow
import yaml
from hydra.utils import get_class
from cellsmap.util.dataset_io import get_model_info
import torch
from typing import List

def load_mlflow_model(
    run_id: str,
    save_path: str,
    checkpoint_path="checkpoints/val/loss/best.ckpt",
    config_path='config/train.yaml',
    tracking_uri: str = "https://production.int.allencell.org/mlflow/",
):
    mlflow.set_tracking_uri(tracking_uri)
    if (save_path / checkpoint_path).exists():
        print("Checkpoint exists! Skipping download...")
    else:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=tracking_uri, artifact_path=checkpoint_path, dst_path=save_path
        )

    if (save_path / config_path).exists():
        print("Config exists! Skipping download...")
    else:
        mlflow.artifacts.download_artifacts(
            run_id=run_id, tracking_uri=tracking_uri, artifact_path=config_path, dst_path=save_path
        )

    with open (save_path / config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_class = get_class(config['model']['_target_'])
    model = model_class.load_from_checkpoint(save_path / checkpoint_path)

    return model

def generate_from_coords(model_name, coords:List[List[float]], n_noise_samples=1, average=False):
    mlflow_id = get_model_info(model_name)['mlflow_id']
    model = load_mlflow_model(mlflow_id, save_path='models')

    coords  = torch.from_numpy(coords).float()

    if torch.cuda.is_available():
        coords = coords.to("cuda")
        model = model.to("cuda")
    walk_img = model.generate_from_latent(coords, n_noise_samples=n_noise_samples, average=average, save=False)
    return walk_img





