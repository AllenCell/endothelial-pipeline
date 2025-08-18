def main() -> None:
    from src.endo_pipeline.io import (
        get_output_path,
        load_dataset_position_as_dask_array,
        load_segmentation,
    )
    from src.endo_pipeline.manifests import (
        get_segmentation_location_for_dataset,
        load_segmentation_manifest,
    )

    # Becky: I would say 20250326 (15 dyn) is probably the overall
    # most ideal dataset. The recent no flow dataset (20250728) is
    # also quite good it just has some quirks around the feedings.

    dataset_name = "20250326_20X"
    position = 0
    timeframe = 276

    out_dir = get_output_path(__file__)

    # panel of raw nuclei brightfield
    # panel of nuclei brightfield std
    # panel of labelfree nuclei prediction
    # panel of raw max project
    # panel of hysteresis thresholding
    # panel of initial cdh5 segmentations
    # panel of merged cdh5 segmentations
    # panel of labelfree nuclei-refined cdh5 segmentations

    dim_order = get_default_dim_order()

    nuc_manifest = load_segmentation_manifest("nuclear_labelfree")
    nuc_location = get_segmentation_location_for_dataset(nuc_manifest, dataset_name, position, T)
    nuc_seg = load_segmentation(nuc_location)

    cdh5_manifest = load_segmentation_manifest("cdh5_classic")
    cdh5_location = get_segmentation_location_for_dataset(cdh5_manifest, dataset_name, position, T)
    cdh5_seg = load_segmentation(cdh5_location)

    raw_img = load_dataset_position_as_dask_array(
        dataset_name=dataset_name,
        position=position,
        channels=channel_names,
        time_start=T,
        time_end=T,
    )
    raw_MIP = raw_img.max(axis=dim_order.index("Z"), keepdims=True).compute()
