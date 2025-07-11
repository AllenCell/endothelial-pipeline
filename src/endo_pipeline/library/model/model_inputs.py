from pathlib import Path

import pandas as pd

from src.endo_pipeline.configs import DatasetConfig

ZARR_BF_CHANNEL = 1  # Brightfield channel index for Zarr files


def generate_zarr_csv(
    dataset_config: DatasetConfig, save_path: Path, resolution_level: int = 1
) -> Path:
    """Generate a CSV file with path to Zarr files for the given dataset."""
    # generate csv with paths to zarr files
    # this replaces the call to get_zarr_path from dataset_io
    zarr_path_list = list(Path(dataset_config.zarr_path).glob("*.zarr"))
    zarr_path_dict = {}
    for path in zarr_path_list:
        zarr_path_dict[path.name] = str(path)

    df = pd.DataFrame({"path": sorted(zarr_path_dict.values())})
    df["channel"] = ZARR_BF_CHANNEL
    df["resolution"] = resolution_level
    data_path = str(save_path / "dataset.csv")
    df.to_csv(data_path, index=False)
    return data_path


def generate_overrides_for_model_eval(
    user_overrides: dict,
    save_path: str,
    data_path: str,
    ckpt_path: str,
    dataset_name: str,
    model_name: str,
) -> dict:
    """
    Generate overrides for the CytoDLModel configuration
    for evaluating model `model_name` on crops of
    images from dataset `dataset_name`.
    """
    overrides = {
        # train and val dataloaders are unnecessary for prediction
        # and might be slow to instantiate (e.g. if they cache data)
        "data.train_dataloaders": None,
        "data.val_dataloaders": None,
        "data.predict_dataloaders.num_workers": 128,
        "data.predict_dataloaders.dataset.csv_path": data_path,
        "paths.output_dir": save_path,
        # change checkpoint path to the one downloaded from mlflow
        "checkpoint.ckpt_path": ckpt_path,
        "checkpoint.strict": True,
        "callbacks": None,
        "callbacks.prediction_saver": {
            "_target_": "cyto_dl.callbacks.tabular_saver.SaveTabularData",
            "save_dir": save_path,
            "meta_keys": [
                "T",
                "start_y",
                "start_x",
                "filename_or_obj",
            ],
            "save_suffix": f"{dataset_name}_{model_name}_features",
        },
    }
    overrides.update(user_overrides)
    return overrides
