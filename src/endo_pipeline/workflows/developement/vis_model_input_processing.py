# %%
from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from src.endo_pipeline.io import load_zarr_as_dask_array
from src.endo_pipeline.io.output import get_output_path
from src.endo_pipeline.library.visualize.model_inputs.image_processing_steps import (
    process_brightfield,
    process_cdh5,
)
from src.endo_pipeline.library.visualize.model_inputs.plot import visualize_images_with_histograms

# %%
POSITION = 0
TIMEPOINT = 0

datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
# datasets = ["20241016"] # Uncomment this line to test with a single dataset

# %% Brightfield Visualization
for dataset in datasets:
    config = load_dataset_config(dataset)

    save_dir = get_output_path("model_input_visualization", "brightfield")
    zarr_file = get_zarr_file_for_position(config, POSITION)
    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=TIMEPOINT, level=1, squeeze=True
    )

    bf_stack_float32_computed, standard_dev_proj, clipped_im, normalized_im = process_brightfield(
        bf_stack
    )

    visualize_images_with_histograms(
        [
            ("BF Slice", bf_stack_float32_computed[15]),
            ("Std Dev Projection", standard_dev_proj),
            ("Clipped Std Dev Projection", clipped_im),
            ("Z-score Normalized Image", normalized_im),
        ],
        save_dir=save_dir,
        fname_prefix=f"{dataset}_P{POSITION}_T{TIMEPOINT}_BF",
    )

# %% CDH5 Visualization
for dataset in datasets:
    config = load_dataset_config(dataset)

    save_dir = get_output_path("model_input_visualization", "cdh5")
    zarr_file = get_zarr_file_for_position(config, POSITION)
    cdh5_stack = load_zarr_as_dask_array(
        zarr_file, channels=["EGFP"], timepoints=TIMEPOINT, level=1, squeeze=True
    )
    
    cdh5_stack_float32_computed, max_proj_im, scaled_im = process_cdh5(cdh5_stack)

    visualize_images_with_histograms(
        [
            ("CDH5 Slice", cdh5_stack_float32_computed[15]),
            ("Max Projection", max_proj_im),
            ("Scaled, Clipped Image", scaled_im),
        ],
        save_dir=save_dir,
        fname_prefix=f"{dataset}_P{POSITION}_T{TIMEPOINT}_CDH5",
    )
# %%
