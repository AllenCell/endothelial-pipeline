from typing import Literal, Sequence

import numpy as np
import pandas as pd

from endo_pipeline.configs import load_model_config
from endo_pipeline.configs.dataset_io import extract_P
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.integration import track_integration
from endo_pipeline.library.model.eval_model import (
    apply_model_on_array,
    load_model_for_inference_from_array_input,
)
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.process.get_images import get_zarr_img_for_dataset
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings import DIFFAE_MODEL_EVAL_FINETUNE_CONFIG


def apply_model_on_array_test(
    dataset_name: str = "20241120_20X",
    model_name: str = "diffae_04_10",
    run_name: str | None = None,
    eval_config: dict | None = None,
    start_x: int = 0,
    start_y: int = 0,
    crop_size_x: int = 128,
    crop_size_y: int = 128,
) -> np.ndarray:

    # load an example image
    img = get_zarr_img_for_dataset(dataset_name, 0, resolution_level=1)
    dim_order = "TCZYX"
    img_arr = img.get_image_dask_data(dim_order, T=0)
    crop_example = (
        slice(None),
        slice(start_x, start_x + crop_size_x),
        slice(start_y, start_y + crop_size_y),
    )
    img_arr_crop_bf = img_arr[(0, slice(1, 2), *crop_example)].compute()

    # load the diffae model and modify the config to accept array inputs
    diffae_model = load_model_for_inference_from_array_input(
        model_name, run_name=None, eval_config=eval_config, save_config_locally=True
    )

    # run th model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf)

    return cytodl_output


def get_image_crop_for_model(
    dataset_name: str = "20241120_20X",
    position: int = 0,
    timeframe_list: Sequence = [0],
    start_x_list: Sequence = [0],
    start_y_list: Sequence = [0],
    crop_size_x: int = 128,
    crop_size_y: int = 128,
    resolution_level: Literal[0, 1] = 1,
) -> list[np.ndarray]:
    # load image
    img = get_zarr_img_for_dataset(dataset_name, position, resolution_level)
    dim_order = "TCZYX"
    img_arr = img.get_image_dask_data(dim_order)
    # make slice objects for cropping
    crop_list = []
    for timeframe, start_x, start_y in zip(timeframe_list, start_x_list, start_y_list):
        crop_list.append(
            (
                timeframe,
                slice(1, 2),
                slice(None),
                slice(start_x, start_x + crop_size_x),
                slice(start_y, start_y + crop_size_y),
            )
        )
    img_arr_crop_bf_list = []
    for crop in crop_list:
        img_arr_crop_bf_list.append(img_arr[*crop].compute())

    return img_arr_crop_bf_list


def preprocess_column_names_for_model(df_precomp: pd.DataFrame) -> pd.DataFrame:
    if not df_precomp["position"].astype(str).str.isdigit().all():
        df_precomp["position"] = df_precomp["position"].transform(extract_P)
    if "frame_number" not in df_precomp.columns:
        df_precomp["frame_number"] = df_precomp["image_index"]
    return df_precomp


if __name__ == "__main__":

    ds = "20241120_20X"
    position = 0

    kind = "cell"
    df_kind = {"cell": 0, "grid": 1}

    # load the manifest for the given dataset and filter for the given position
    df_precomp = track_integration.get_preprocessed_manifests_and_km_bounds(ds)[df_kind[kind]]

    df_precomp = preprocess_column_names_for_model(df_precomp)  # type:ignore
    df_precomp = df_precomp.query("position == @position")
    crop_size_x = int(sequence_to_scalar(df_precomp.crop_size_x))
    crop_size_y = int(sequence_to_scalar(df_precomp.crop_size_y))
    model_name = sequence_to_scalar(df_precomp["model_name"])
    run_name = None
    eval_config = load_model_config(DIFFAE_MODEL_EVAL_FINETUNE_CONFIG)

    samples = df_precomp.sample(n=1, random_state=42)

    # load the images for the given dataset and position and crop them according to the manifest
    img_arr_crop_bf_list = get_image_crop_for_model(
        dataset_name=ds,
        position=position,
        timeframe_list=samples.frame_number.values.tolist(),
        start_x_list=samples.start_x.values.tolist(),
        start_y_list=samples.start_y.values.tolist(),
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
        resolution_level=1,
    )

    # load the diffae model and modify the config to accept array inputs
    diffae_model = load_model_for_inference_from_array_input(
        model_name, run_name=None, eval_config=eval_config, save_config_locally=True
    )

    # run the model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf_list)
    feats_calc, affine = zip(*cytodl_output)

    feat_columns = [f"feat_{i}" for i in range(8)]
    feats = samples[feat_columns].astype(float)

    print(ds, position)
    print(feats)
    print(feats_calc)
