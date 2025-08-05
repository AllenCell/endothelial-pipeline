# %%
from src.endo_pipeline.configs import (
    get_datasets_in_collection,
    get_zarr_file_for_position,
    load_dataset_config,
)
from src.endo_pipeline.io import load_zarr_as_dask_array
from src.endo_pipeline.io.output import get_output_path
from src.endo_pipeline.library.process.z_stack_selection import (
    append_projection_outputs,
    get_center_plane_for_position,
    plot_bottom_top_slices,
    plot_image_row,
    save_projection_image,
)
from src.endo_pipeline.library.visualize.model_inputs.image_processing_steps import (
    process_brightfield,
    process_cdh5,
)
from src.endo_pipeline.library.visualize.model_inputs.plot import visualize_images_with_histograms

# %%
POSITION = 0
TIMEPOINT = 0

datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
# datasets = ["20241016_20X"]  # Uncomment this line to test with a single dataset

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

# %% Visualize projection made from varied slices (Bottom, Top, Center, Full Stack)
for dataset in datasets:
    config = load_dataset_config(dataset)
    save_dir = get_output_path("model_input_visualization", "z_stack_selection")
    zarr_file = get_zarr_file_for_position(config, POSITION)

    bf_stack = load_zarr_as_dask_array(
        zarr_file, channels=["BF"], timepoints=TIMEPOINT, level=1, squeeze=True
    )
    cdh5_stack = load_zarr_as_dask_array(
        zarr_file, channels=["EGFP"], timepoints=TIMEPOINT, level=1, squeeze=True
    )

    center_slice = get_center_plane_for_position(config, POSITION)
    slices = [[0, 16], [9, 24], [center_slice - 5, center_slice + 10], [0, bf_stack.shape[0] - 1]]

    bf_images: list = []
    bf_titles: list[str] = []
    bf_bottom_slice: list = []
    bf_top_slice: list = []

    cdh5_images: list = []
    cdh5_titles: list[str] = []
    cdh5_bottom_slice: list = []
    cdh5_top_slice: list = []

    for zslice in slices:
        slice_str = f"{zslice[0]}_{zslice[1]}"

        # BF
        append_projection_outputs(
            bf_stack,
            zslice,
            process_brightfield,
            bf_images,
            bf_titles,
            bf_bottom_slice,
            bf_top_slice,
        )
        bf_fname = save_dir / f"{dataset}_P{POSITION}_T{TIMEPOINT}_BF_max_proj_{slice_str}.png"
        save_projection_image(bf_images[-1], bf_fname)

        # CDH5
        append_projection_outputs(
            cdh5_stack,
            zslice,
            process_cdh5,
            cdh5_images,
            cdh5_titles,
            cdh5_bottom_slice,
            cdh5_top_slice,
        )
        cdh5_fname = save_dir / f"{dataset}_P{POSITION}_T{TIMEPOINT}_CDH5_max_proj_{slice_str}.png"
        save_projection_image(cdh5_images[-1], cdh5_fname)

    plot_image_row(bf_images, bf_titles, dataset, save_dir, row_title="BF slice")
    plot_image_row(cdh5_images, cdh5_titles, dataset, save_dir, row_title="CDH5 slice")
    plot_bottom_top_slices(bf_bottom_slice, bf_top_slice, bf_titles, dataset, save_dir, label="BF")
    plot_bottom_top_slices(
        cdh5_bottom_slice, cdh5_top_slice, cdh5_titles, dataset, save_dir, label="CDH5"
    )
# %%
