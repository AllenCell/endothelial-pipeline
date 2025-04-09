import json
import fire
import torch
from typing import Dict, Union
from cellsmap.util.dataset_io import get_model_info, get_dataset_info, get_zarr_path
from cyto_dl.api import CytoDLModel
from pathlib import Path
from cellsmap.model_features.utils.mlflow_utils import download_model, download_mlflow_artifact
from cellsmap.util.set_ouput import get_output_path
import pandas as pd
from cellsmap.util.manifest_preprocessing import save_file_to_fms


def get_cytodl_commit_hash(run_id: str, model_path: Path) -> str:
    """
    Extract commit hash from the requirements file uploaded to mlflow
    
    Parameters
    ----------
    run_id: str
        The run ID of the MLflow run.
    model_path: Path
        The path where the downloaded model artifacts are saved.
    """
    artifact_path  = 'requirements/eval-requirements.txt'
    download_mlflow_artifact(run_id, artifact_path, model_path)
    with open(model_path/artifact_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'git+' in line and 'cyto-dl' in line:
            commit_hash = line.split('git+')[1].split('#egg')[0].split('/')[-1]
            return commit_hash
    raise ValueError('No commit hash found in requirements.txt')

def generate_overrides(save_path: str, data_path: str, ckpt_path: str, dataset_name: str, model_name: str) -> Dict:
    overrides = {
        # train and val dataloaders are unnecessary for prediction and might be slow to instantiate (e.g. if they cache data)
        'data.train_dataloaders': None,
        'data.val_dataloaders': None,
        'data.predict_dataloaders.num_workers': 128,
        'data.predict_dataloaders.dataset.csv_path': data_path,
        'paths.output_dir': save_path,
        # change checkpoint path to the one downloaded from mlflow
        'checkpoint.ckpt_path': ckpt_path,
        'checkpoint.strict': True,
        'callbacks': None,
        'callbacks.prediction_saver': {
            "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
            "save_dir": str(save_path),
            "meta_keys": ["T", 'start_y', 'start_x', 'filename_or_obj'],
            "suffix": f"{dataset_name}_{model_name}_features"
        },
    }
    return overrides

def generate_zarr_csv(dataset_name: str, save_path: str, resolution: int=0):
    # generate csv with paths to zarr files
    channel = get_dataset_info(dataset_name)['brightfield_channel_index']
    df = pd.DataFrame({
        'path': sorted(Path(get_zarr_path(dataset_name)).glob('*.zarr'))
    })
    df['channel'] = channel 
    df['resolution'] = resolution
    data_path = str(save_path / 'dataset.csv')
    df.to_csv(data_path, index=False)
    return data_path

def update_prediction_with_meta(dataset_name: str, model_name: str, mlflow_id: str, save_path: Path):
    # add model and dataset information to prediction file
    prediction_path = save_path/f"predict_{dataset_name}_{model_name}_features.parquet"
    pred_df = pd.read_parquet(prediction_path)
    pred_df['dataset'] = dataset_name
    pred_df['model_name'] = model_name
    pred_df['mlflow_id'] = mlflow_id
    pred_df.rename(columns={'filename_or_obj': 'zarr_path'}, inplace=True)
    pred_df.to_parquet(prediction_path)
    return prediction_path

def apply_model(model_name:str, dataset_name: str, resolution:int=0, overrides:Union[str, Dict]={}):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Please run on a GPU machine.')

    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    elif not isinstance(overrides, dict):
        raise ValueError('Overrides must be a dictionary or a string')
    
    # download model from mlflow
    mlflow_id = get_model_info(model_name)['mlflow_run_id']
    model_path = Path(get_output_path(f'models/{model_name}'))
    path_dict = download_model(mlflow_id, model_path)

    save_path = model_path/dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    # load model
    model = CytoDLModel()
    model.load_config_from_file(path_dict['config_path'])

    # create zarr dataset
    data_path = generate_zarr_csv(dataset_name, save_path, resolution)

    # apply overrides
    overrides = generate_overrides(
        save_path=save_path,
        data_path=data_path,
        ckpt_path=path_dict['checkpoint_path'],
        dataset_name=dataset_name,
        model_name=model_name
    )
    model.override_config(overrides)
    model.predict()

    prediction_path = update_prediction_with_meta(
        dataset_name=dataset_name,
        model_name=model_name,
        mlflow_id=mlflow_id,
        save_path=save_path
    )
    commit_hash = get_cytodl_commit_hash(mlflow_id, model_path)

    save_file_to_fms(prediction_path, dataset_name, commit_hash, misc_notes='', mlflow_run_id=mlflow_id)

if __name__ == '__main__':
    fire.Fire(apply_model)