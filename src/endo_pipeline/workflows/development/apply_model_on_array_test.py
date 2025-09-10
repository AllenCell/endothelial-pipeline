from typing import Sequence

import numpy as np

from endo_pipeline.configs.dataset_io import extract_P
from endo_pipeline.library.analyze.integration import track_integration
from endo_pipeline.library.model.apply_model import apply_model_on_array, get_model_for_array_inputs
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.library.process.get_images import get_zarr_img_for_dataset


def apply_model_on_array_test(
    dataset_name: str = "20241120_20X",
    model_name: str = "diffae_04_10",
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
    diffae_model = get_model_for_array_inputs(model_name, save_config_locally=True)

    # run th model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf)

    return cytodl_output


def get_image_crop_for_model(
    dataset_name: str = "20241120_20X",
    position: int = 0,
    timeframe_list: Sequence = [0],
    start_x_list: Sequence = [0],
    start_y_list: Sequence = [0],
    crop_size_x: Sequence = 128,
    crop_size_y: Sequence = 128,
) -> list[np.ndarray]:
    # load image
    img = get_zarr_img_for_dataset(dataset_name, position, resolution_level=1)
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


if __name__ == "__main__":

    ds = "20241120_20X"
    position = 0
    model_name = "diffae_04_10"

    df_precomp = track_integration.get_preprocessed_manifests_and_km_bounds(dataset_name=ds)[1]
    df_precomp["position"] = df_precomp["position"].transform(extract_P)
    samples = df_precomp.query("position==@position").sample(n=12, random_state=0)

    crop_size_x = sequence_to_scalar(samples.crop_size_x)
    crop_size_y = sequence_to_scalar(samples.crop_size_y)

    img_arr_crop_bf_list = get_image_crop_for_model(
        dataset_name=ds,
        position=position,
        timeframe_list=samples.frame_number,
        start_x_list=samples.start_x,
        start_y_list=samples.start_y,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
    )

    # load the diffae model and modify the config to accept array inputs
    diffae_model = get_model_for_array_inputs(model_name, save_config_locally=True)

    # run th model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf_list)
    feats_calc, affine = zip(*cytodl_output)

    feat_columns = [f"feat_{i}" for i in range(8)]
    feats = samples[feat_columns].astype(float)

    print(feats, feats_calc)

    # for idx, row in samples.iterrows():
    #     features, affine = apply_model_on_array_test(
    #         dataset_name=ds,
    #         start_x=row.start_x,
    #         crop_size_x=row.crop_size_x,
    #         start_y=row.start_y,
    #         crop_size_y=row.crop_size_y,
    #     )
    #     print(ds, idx, features, row["feat_0":"feat_7"].values.astype(float))

    # for idx, row in samples.iterrows():
    #     features, affine = apply_model_on_array_test(
    #         dataset_name=ds,
    #         start_x=row.start_x,
    #         crop_size_x=row.crop_size_x,
    #         start_y=row.start_y,
    #         crop_size_y=row.crop_size_y,
    #     )
    #     print(ds, idx, features)
