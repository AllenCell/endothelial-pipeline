# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_available_zarr_files,
    get_datasets_in_collection,
    load_dataset_config,
    save_dataset_config,
)
from endo_pipeline.io.input import load_zarr_as_dask_array
from endo_pipeline.library.process.bf_timepoint_outlier import detect_outliers

# %% LOAD DATA
# datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
datasets = [
    "20250402_20X",
    "20250409_20X",
    "20250428_20X",
    "20250604_20X",
    "20250611_20X",
    "20250618_20X",
    "20250714_20X",
    "20250716_20X",
    "20250728_20X",
    "20250806_20X",
]

for dataset_name in datasets:

    dataset_config = load_dataset_config(dataset_name)

    if manual_tp_annotations is None:
        print("No manual annotations found.")
        continue

    auto_tp_annotations = {
        TimepointAnnotation.BF_SCOPE_ERROR: {
            position: [] for position in dataset_config.zarr_positions
        },
        TimepointAnnotation.BF_TEMP_ARTIFACT: {
            position: [] for position in dataset_config.zarr_positions
        },
    }

    # Iterate over positions and detect outliers
    for position in dataset_config.zarr_positions:
        bf_scope_error, bf_temp_artifact = detect_outliers(dataset_config, position, visualize=True)

        # Append the detected outliers to the corresponding lists
        auto_tp_annotations[TimepointAnnotation.BF_SCOPE_ERROR][position].extend(bf_scope_error)
        auto_tp_annotations[TimepointAnnotation.BF_TEMP_ARTIFACT][position].extend(bf_temp_artifact)

    # Iterate over positions in manual_tp_annotations
    print(dataset_name)
    manual_tp_annotations = dataset_config.timepoint_annotations
    # Initialize overall statistics
    overall_detected = 0
    overall_manual = 0
    overall_missed = 0

    for position in dataset_config.zarr_positions:
        # Combine manual_tp_annotations for the current position
        manual_scope_error = set(
            manual_tp_annotations.get(TimepointAnnotation.BF_SCOPE_ERROR, {}).get(position, [])
        )
        manual_temp_artifact = set(
            manual_tp_annotations.get(TimepointAnnotation.BF_TEMP_ARTIFACT, {}).get(position, [])
        )
        manual_tps = manual_scope_error | manual_temp_artifact

        # Combine auto_tp_annotations for the current position
        auto_tps = set(
            auto_tp_annotations[TimepointAnnotation.BF_SCOPE_ERROR].get(position, [])
        ) | set(auto_tp_annotations.get(TimepointAnnotation.BF_TEMP_ARTIFACT, {}).get(position, []))

        # Check if all manual_tps are accounted for in auto_tps
        missing_tps = manual_tps - auto_tps
        if missing_tps:
            print(f"P{position}: Missing timepoints in auto_tp_annotations: {missing_tps}")

        total_detected, total_manual, missed = len(auto_tps), len(manual_tps), len(missing_tps)

        print(
            f"P{position}: \
                Total Detected: {total_detected}, Total Manual: {total_manual}, Missed: {missed}"
        )

        overall_detected += total_detected
        overall_manual += total_manual
        overall_missed += missed

    print("\nOverall Statistics:")
    print(f"Total Detected: {overall_detected}")
    print(f"Total Manual: {overall_manual}")
    print(f"Total Missed: {overall_missed}")
    # %%
