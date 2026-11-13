import logging

import pandas as pd

from endo_pipeline.cli import DEMO_MODE, PatchType
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, join_sorted_strings, upload_file_to_fms
from endo_pipeline.library.analyze.vector_field_estimation import get_valid_flow_field_column_names
from endo_pipeline.manifests import (
    DataframeLocation,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_WIDTHS_DYNAMICS,
    KERNEL_BANDWIDTHS_DYNAMICS,
)
from endo_pipeline.settings.flow_field_dataframes import (
    FMS_ANNOTATION_NOTES_BOOTSTRAPPING,
    FMS_ANNOTATION_NOTES_FIXED_POINTS,
    FMS_ANNOTATION_NOTES_VECTOR_FIELD,
)
from endo_pipeline.settings.manifest_names import (
    BOOTSTRAPPING_MANIFEST_NAMES,
    FIXED_POINT_MANIFEST_NAMES,
    VECTOR_FIELD_MANIFEST_NAMES,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)


def get_param_sweep_run_name(bin_width: float, kernel_bandwidth: float) -> str:
    """
    Generate a name for the parameter sweep based on the bin and kernel scales.

    Parameters
    ----------
    bin_width
        Bin width used for the parameter sweep.
    kernel_bandwidth
        Kernel bandwidth used for the parameter sweep.

    Returns
    -------
    :
        Name representing the parameter sweep.
    """
    bin_width_as_str = str(round(bin_width, 3)).replace(".", "pt")
    kernel_bandwidth_as_str = str(round(kernel_bandwidth, 3)).replace(".", "pt")

    tag = f"bw{bin_width_as_str}_kb{kernel_bandwidth_as_str}"
    run_name_for_sweep_condition = f"{DEFAULT_MODEL_RUN_NAME}_{tag}"

    return run_name_for_sweep_condition


def get_binwidth_bandwidth_dict(
    bin_width: float, kernel_bandwidth: float
) -> tuple[dict[Column.DiffAEData, float], dict[Column.DiffAEData, float]]:
    """
    Get dictionaries of bin widths and kernel bandwidths for the provided parameters propagating the
    provided bin width and kernel bandwidth to all entries in `BIN_WIDTHS_DYNAMICS` and
    `KERNEL_BANDWIDTHS_DYNAMICS`.

    Parameters
    ----------
    bin_width
        Bin width to be used.
    kernel_bandwidth
        Kernel bandwidth to be used.

    Returns
    -------
    :
        Dictionaries of bin widths and kernel bandwidths.
    """
    bin_widths = dict.fromkeys(BIN_WIDTHS_DYNAMICS.keys(), bin_width)
    kernel_bandwidths = dict.fromkeys(KERNEL_BANDWIDTHS_DYNAMICS.keys(), kernel_bandwidth)
    return bin_widths, kernel_bandwidths


def upload_bootstrapped_fixed_points_to_fms(
    output_file_name_suffix: str,
    dataset_name: str,
    patch_type: PatchType = "grid_based",
    overwrite_fmsids: bool = True,
):
    name_prefix = BOOTSTRAPPING_MANIFEST_NAMES[patch_type]
    name_suffix = "demo" if DEMO_MODE else ""
    bootstrap_results_manifest_name = "_".join(
        filter(None, [name_prefix, name_suffix, output_file_name_suffix])
    )

    dataset_config = load_dataset_config(dataset_name)
    bootstrap_results_manifest = load_dataframe_manifest(bootstrap_results_manifest_name)
    location = bootstrap_results_manifest.locations.get(dataset_name, DataframeLocation())
    if location.path is None:
        logger.warning(
            "No local path found for dataset %s in manifest %s",
            dataset_name,
            bootstrap_results_manifest_name,
        )
        return
    if overwrite_fmsids is False and location.fmsid is not None:
        logger.info(
            "Skipping upload for dataset %s in manifest %s as results already exist on FMS",
            dataset_name,
            bootstrap_results_manifest_name,
        )
        return
    output_save_path = location.path

    # Upload to FMS (internal only)
    annotations = build_fms_annotations(
        dataset_config,
        model_manifest=load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME),
        run_name=DEFAULT_MODEL_MANIFEST_NAME,
        additional_notes=FMS_ANNOTATION_NOTES_BOOTSTRAPPING,
    )
    fmsid = upload_file_to_fms(output_save_path, annotations=annotations, file_type="parquet")
    location.fmsid = fmsid
    location.path = None

    # Add dataframe location to dataframe manifest and save
    bootstrap_results_manifest.locations[dataset_name] = location
    save_dataframe_manifest(bootstrap_results_manifest)


def upload_flow_fields_to_fms(
    output_file_name_suffix: str,
    dataset_name: str,
    patch_type: PatchType = "grid_based",
    overwrite_fmsids: bool = True,
    column_names: list[str] | list[Column.DiffAEData] | None = None,
):
    logger = logging.getLogger(__name__)

    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)

    column_names = get_valid_flow_field_column_names(column_names)

    name_suffix = join_sorted_strings(column_names)
    vector_field_dataframe_manifest_name = "_".join(
        filter(
            None, [VECTOR_FIELD_MANIFEST_NAMES[patch_type], name_suffix, output_file_name_suffix]
        )
    )
    fixed_points_dataframe_manifest_name = "_".join(
        filter(None, [FIXED_POINT_MANIFEST_NAMES[patch_type], name_suffix, output_file_name_suffix])
    )

    vector_field_dataframe_manifest = load_dataframe_manifest(vector_field_dataframe_manifest_name)
    fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)

    vector_field_location = vector_field_dataframe_manifest.locations.get(
        dataset_name, DataframeLocation()
    )
    fixed_points_location = fixed_points_dataframe_manifest.locations.get(
        dataset_name, DataframeLocation()
    )

    if (vector_field_location.path is None) or (fixed_points_location.path is None):
        logger.warning(
            "Vector field or fixed points location for dataset %s is not available.",
            dataset_name,
        )
        return
    vector_field_for_dataset = pd.read_parquet("/" + vector_field_location.path.as_posix())
    fixed_points_for_dataset = pd.read_parquet("/" + fixed_points_location.path.as_posix())

    dataset_config = load_dataset_config(dataset_name)

    for manifest, dataframe, _, additional_notes in [
        (
            vector_field_dataframe_manifest,
            vector_field_for_dataset,
            VECTOR_FIELD_MANIFEST_NAMES[patch_type],
            FMS_ANNOTATION_NOTES_VECTOR_FIELD % len(column_names),
        ),
        (
            fixed_points_dataframe_manifest,
            fixed_points_for_dataset,
            FIXED_POINT_MANIFEST_NAMES[patch_type],
            FMS_ANNOTATION_NOTES_FIXED_POINTS % len(column_names),
        ),
    ]:
        # Skip if dataframe is None.
        if dataframe is None:
            continue

        # Create location object with output path
        location = manifest.locations.get(dataset_name, DataframeLocation())

        if location.path is None:
            logger.warning(
                "No local path found for dataset %s in manifest %s",
                dataset_name,
                manifest.name,
            )
            return
        if overwrite_fmsids is False and location.fmsid is not None:
            logger.info(
                "Skipping upload for dataset %s in manifest %s as results already exist on FMS",
                dataset_name,
                manifest.name,
            )
            return

        save_path = location.path

        # Upload to FMS (internal only)
        annotations = build_fms_annotations(
            dataset_config,
            model_manifest=model_manifest,
            run_name=DEFAULT_MODEL_RUN_NAME,
            additional_notes=additional_notes,
        )
        fmsid = upload_file_to_fms(save_path, annotations=annotations, file_type="parquet")
        location.fmsid = fmsid
        location.path = None

        # Add dataframe location to dataframe manifest and save
        manifest.locations[dataset_name] = location
        save_dataframe_manifest(manifest)
