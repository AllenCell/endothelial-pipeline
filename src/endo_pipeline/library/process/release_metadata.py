"""Methods for adding metadata for data release."""

from pathlib import Path
from typing import Any

from endo_pipeline.configs import (
    ChannelName,
    DatasetConfig,
    get_available_dataset_names,
    get_datasets_in_collection,
    load_dataset_config,
)
from endo_pipeline.manifests import (
    DataframeManifest,
    ImageManifest,
    ModelManifest,
    get_dataframe_location_for_dataset,
    get_image_location_for_dataset,
    get_model_location_for_run,
)
from endo_pipeline.settings.image_data import (
    NUM_ZSLICES,
    IMG_SHAPE_RESOLUTION_0_3i_X,
    IMG_SHAPE_RESOLUTION_0_3i_Y,
    PIXEL_SIZE_3i_20x,
    Z_STEP_SIZE_ACTUAL_3i_20x,
)

CELL_LINES_METADATA = {
    "AICS-126 cl. 41": "Vascular endothelial VE-cadherin (CD144-sorted)",
    "AICS-126 cl. 41 CD31-sorted": "Vascular endothelial VE-cadherin (CD31-sorted)",
    "AICS-177 cl. 26": "Vascular endothelial VE-cadherin Exon 3 Deletion (CD31-sorted)",
}
"""Metadata mapping for cell lines."""


CHANNEL_WAVELENGTH_METADATA = {
    ChannelName.EGFP: "488 nm excitation laser (LuxX Diode laser series)",
    ChannelName.BF: "740 nm LED (Lambda TLED+, Sutter Instruments)",
    ChannelName.NucViolet: "405 nm excitation laser (LuxX Diode laser series)",
    ChannelName.SOX17: "561 nm excitation laser (LuxX Diode laser series)",
    ChannelName.NR2F2: "640 nm excitation laser (LuxX Diode laser series)",
    ChannelName.DAPI: "???",
}
"""Metadata mapping for channel wavelength."""


CHANNEL_CONTENT_METADATA = {
    ChannelName.EGFP: "mEGFP-VE-cadherin fluorescence emission",
    ChannelName.BF: "Transmitted light signal (brightfield)",
    ChannelName.NucViolet: "Nuclear Violet LCS1 stain emission",
    ChannelName.SOX17: "Anti-Sox17 Clone OTI3B10 Mouse Monoclonal Antibody, Goat anti-Mouse IgG (H+L) Alexa Fluor™ Plus 555 emission",
    ChannelName.NR2F2: "Anti-NR2F2 Rabbit Monoclonal Antibody, Goat anti-Rabbit IgG (H+L) Alexa Fluor™ Plus 647 emission",
    ChannelName.DAPI: "???",
}
"""Metadata mapping for channel content."""

CHANNEL_LASER_POWER_METADATA = {
    ChannelName.EGFP: "3.30",
    ChannelName.BF: "Intensity histogram of brightfield images was adjusted to peak at around ~14,000 in grayscale value",
    ChannelName.NucViolet: "0.8",
    ChannelName.SOX17: "11",
    ChannelName.NR2F2: "15",
    ChannelName.DAPI: "???",
}
"""Metadata mapping for channel laser power."""


def build_image_manifest_release_metadata(
    manifest: ImageManifest, dataset_config: DatasetConfig, position: int
) -> dict:
    """Build release metadata for image manifest."""

    s3uri = get_image_location_for_dataset(manifest, dataset_config, position).s3uri

    assert s3uri is not None

    metadata: dict[str, Any] = {}

    add_file_metadata(metadata, s3uri, manifest.name)
    add_general_metadata(metadata)
    add_dataset_metadata(metadata, dataset_config)
    add_image_metadata(metadata, dataset_config, manifest.name)

    return metadata


def build_dataframe_manifest_release_metadata(
    manifest: DataframeManifest, location_key: str
) -> dict:
    s3uri = get_dataframe_location_for_dataset(manifest, location_key).s3uri

    assert s3uri is not None

    metadata: dict[str, Any] = {}

    add_file_metadata(metadata, s3uri, manifest.name)
    add_general_metadata(metadata)

    if location_key in get_available_dataset_names():
        dataset_config = load_dataset_config(location_key)
        add_dataset_metadata(metadata, dataset_config)

    add_dataframe_metadata(metadata, manifest)

    return metadata


def build_model_manifest_release_metadata(manifest: ModelManifest, location_key: str) -> list[dict]:
    s3uri = get_model_location_for_run(manifest, location_key).s3uri

    assert s3uri is not None
    s3uri = [s3uri] if isinstance(s3uri, str) else s3uri

    metadata_list = []

    for uri in s3uri:
        metadata: dict[str, Any] = {}
        add_file_metadata(metadata, uri, manifest.name)
        add_general_metadata(metadata)
        add_model_metadata(metadata, manifest.name)
        metadata_list.append(metadata)

    return metadata_list


def add_file_metadata(metadata: dict, s3uri: str, manifest_name: str) -> None:
    """Add metadata about the file itself."""

    _, key = s3uri.replace("s3://", "").split("/", 1)

    metadata["File Path"] = s3uri
    metadata["File Name"] = key
    metadata["File Format"] = "".join(Path(key).suffixes)[1:]
    metadata["File Manifest"] = manifest_name


def add_general_metadata(metadata: dict) -> None:
    """Add metadata that applies to all file types."""

    metadata["Study Type"] = "placeholder"
    metadata["Study Description"] = "placeholder"
    metadata["Publication Title"] = "placeholder"
    metadata["Publication DOI"] = "doi: placeholder"


DATASET_COLLECTION_NAMES = {
    "shear_stress": "Shear stress dataset",
    "diffae_model_training": "DiffAE dataset",
    "nuclear_labelfree_model_training": "Nuclear label-free model training dataset",
    "perturbation": "VE-cadherin Exon3Del perturbation dataset",
}
"""Mapping of collection names to dataset display names."""


def add_dataset_metadata(metadata: dict, dataset: DatasetConfig) -> None:
    """Add metadata specific to the dataset."""

    dataset_names = []
    for collection, display_name in DATASET_COLLECTION_NAMES.items():
        if dataset.name in get_datasets_in_collection(collection):
            dataset_names.append(display_name)
    metadata["Dataset"] = ", ".join(dataset_names) if dataset_names else ""

    metadata["Identity"] = dataset.barcode
    metadata["Date"] = dataset.date
    metadata["Original File ID"] = dataset.fmsid
    metadata["Organism"] = "human"
    metadata["Biological entity"] = "WTC-11 hiPSC derived endothelial cells"
    metadata["Cell Line"] = CELL_LINES_METADATA[dataset.cell_lines[0]]
    metadata["Replicate"] = dataset.replicate_number

    shear_stress_regime = " to ".join(r.value for r in dataset.shear_stress_regime)
    metadata["Shear Stress Regime"] = shear_stress_regime

    for index, flow_condition in enumerate(dataset.flow_conditions):
        flow_shear_stress = round(flow_condition.shear_stress)
        flow_start = flow_condition.start if index != 0 else 0
        flow_stop = flow_condition.stop
        metadata[f"Shear Stress {index + 1} (dynes/cm²)"] = flow_shear_stress
        metadata[f"Shear Stress {index + 1} Frame Start"] = flow_start
        metadata[f"Shear Stress {index + 1} Frame Stop"] = flow_stop

    flow_start = dataset.flow_conditions[0].start
    time_interval = dataset.time_interval_in_minutes or 1
    metadata["Shear Duration Prior to Imaging (min)"] = abs(flow_start) * time_interval


def add_image_metadata(metadata: dict, dataset: DatasetConfig, manifest_name: str) -> None:
    """Add image-only metadata."""

    metadata["Data Type"] = "Image Data"

    if manifest_name == "image_zarr":
        metadata["Imaging Method"] = f"{dataset.microscope} spinning disk confocal microscopy"
        metadata["Live or Fixed Cell Sample"] = dataset.live_or_fixed_sample
        metadata["Preparation Method"] = "See method description in publication"
        metadata["Image Acquisition Settings"] = "See method description in publication"
        metadata["Objective"] = dataset.objective
        metadata["Objective Immersion Medium Refraction Index"] = "air RI=1.0"

        for index, channel in enumerate(dataset.channel_names):
            metadata[f"Channel {index} Wavelength"] = CHANNEL_WAVELENGTH_METADATA[channel]
            metadata[f"Channel {index} Image Content"] = CHANNEL_CONTENT_METADATA[channel]
            metadata[f"Channel {index} Laser Power (mW)"] = CHANNEL_LASER_POWER_METADATA[channel]

    metadata["Timelapse Duration (frames)"] = dataset.duration
    metadata["Time Interval (min)"] = dataset.time_interval_in_minutes

    if manifest_name == "image_zarr":
        metadata["Image Type"] = "raw"
    else:
        metadata["Image Type"] = "segmentation"

    if manifest_name == "nuclear_labelfree_seg_zarr":
        metadata["Segmentation Structure"] = "Nuclei"
    elif manifest_name == "cdh5_classic_seg_zarr":
        metadata["Segmentation Structure"] = "VE-cadherin"
    elif manifest_name == "cdh5_seg_validations_zarr":
        metadata["Segmentation Structure"] = "VE-cadherin,Nuclei"

    metadata["Image Format"] = "ome.zarr"

    if manifest_name == "image_zarr":
        metadata["Image Shape (TCZYX)"] = (
            dataset.duration,
            len(dataset.channel_names),
            NUM_ZSLICES,
            IMG_SHAPE_RESOLUTION_0_3i_Y,
            IMG_SHAPE_RESOLUTION_0_3i_X,
        )
    elif manifest_name in ("cdh5_classic_seg_zarr", "nuclear_labelfree_seg_zarr"):
        metadata["Image Shape (TCZYX)"] = (
            dataset.duration,
            1,
            1,
            IMG_SHAPE_RESOLUTION_0_3i_Y,
            IMG_SHAPE_RESOLUTION_0_3i_X,
        )
    elif manifest_name == "cdh5_seg_validations_zarr":
        metadata["Image Shape (TCZYX)"] = (
            dataset.duration // 48,
            8,
            1,
            IMG_SHAPE_RESOLUTION_0_3i_Y,
            IMG_SHAPE_RESOLUTION_0_3i_X,
        )

    metadata["Pixel Size X (microns)"] = PIXEL_SIZE_3i_20x
    metadata["Pixel Size Y (microns)"] = PIXEL_SIZE_3i_20x

    if manifest_name == "image_zarr":
        metadata["Pixel Size Z (microns)"] = Z_STEP_SIZE_ACTUAL_3i_20x


def add_dataframe_metadata(metadata: dict, manifest: DataframeManifest) -> None:
    """Add dataframe-only metadata."""

    metadata["Data Type"] = "Analysis Data"
    metadata["Analysis Workflow"] = manifest.workflow


def add_model_metadata(metadata: dict, manifest_name: str) -> None:
    """Add model-only metadata."""

    metadata["Data Type"] = "Model Checkpoints"

    if manifest_name == "nuc_pred_labelfree":
        metadata["Model Type"] = "Cellpose"
    else:
        metadata["Model Type"] = "DiffAE"
