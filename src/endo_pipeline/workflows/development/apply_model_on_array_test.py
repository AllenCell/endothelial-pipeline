import numpy as np

from endo_pipeline.library.model.apply_model import apply_model_on_array, get_model_for_array_inputs
from endo_pipeline.library.process.get_images import get_zarr_img_for_dataset


def apply_model_on_array_test(
    dataset_name: str = "20241120_20X", model_name: str = "diffae_04_10"
) -> np.ndarray:

    # load an example image
    img = get_zarr_img_for_dataset(dataset_name, 0, resolution_level=1)
    dim_order = "TCZYX"
    img_arr = img.get_image_dask_data(dim_order, T=0)
    crop_example = (slice(None), slice(0, 128), slice(0, 128))
    img_arr_crop_bf = img_arr[(0, slice(1, 2), *crop_example)].compute()

    # load the diffae model and modify the config to accept array inputs
    diffae_model = get_model_for_array_inputs(model_name, save_config_locally=True)

    # run th model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf)

    return cytodl_output


if __name__ == "__main__":
    cytodl_output_example = apply_model_on_array_test()
    for features, affine in cytodl_output_example:
        print(features)
