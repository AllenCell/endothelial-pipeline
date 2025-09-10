import numpy as np

from endo_pipeline.library.analyze.integration import track_integration
from endo_pipeline.library.model.apply_model import apply_model_on_array, get_model_for_array_inputs
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
    crop_example = (slice(None), slice(start_x, crop_size_x), slice(start_y, crop_size_y))
    img_arr_crop_bf = img_arr[(0, slice(1, 2), *crop_example)].compute()

    # load the diffae model and modify the config to accept array inputs
    diffae_model = get_model_for_array_inputs(model_name, save_config_locally=True)

    # run th model on the example image crop
    cytodl_output = apply_model_on_array(diffae_model, img_arr_crop_bf)

    return cytodl_output


if __name__ == "__main__":

    ds = "20241120_20X"

    # cytodl_output_example = apply_model_on_array_test(dataset_name=ds)
    # for features, affine in cytodl_output_example:
    #     print(features)

    # df = track_integration.get_diffae_feats_liveseg_feats_merged_table(dataset_name=ds)

    df_precomp = track_integration.get_preprocessed_manifests_and_km_bounds(dataset_name=ds)[1]

    for idx, row in df_precomp.sample(n=12).iterrows():
        features, affine = apply_model_on_array_test(
            dataset_name=ds,
            start_x=row.start_x,
            crop_size_x=row.crop_size_x,
            start_y=row.start_y,
            crop_size_y=row.crop_size_y,
        )
        print(ds, idx, features)
