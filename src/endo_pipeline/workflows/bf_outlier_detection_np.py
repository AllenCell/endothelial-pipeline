# %%
import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import (
    DatasetConfig,
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
    save_dataset_config,
)
from endo_pipeline.library.process.bf_timepoint_outlier import detect_outliers

# %% LOAD DATA
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
dataset_name = datasets[0]
# for dataset_name in datasets:
dataset_config = load_dataset_config(dataset_name)
tp_annotations = dataset_config.timepoint_annotations
# %%
auto_tp_annotations = {
    TimepointAnnotation.AUTO_BF_SCOPE_ERROR: {
        position: [] for position in dataset_config.zarr_positions
    },
    TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT: {
        position: [] for position in dataset_config.zarr_positions
    },
}
# %% Iterate over positions and detect outliers
for position in dataset_config.zarr_positions:
    bf_scope_error, bf_temp_artifact = detect_outliers(dataset_config, position, visualize=True)

    # Append the detected outliers to the corresponding lists
    auto_tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR][position].extend(bf_scope_error)
    auto_tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT][position].extend(
        bf_temp_artifact
    )

# %% Save the updated dataset configuration with auto-detected annotations
dataset_config.timepoint_annotations = tp_annotations | auto_tp_annotations
save_dataset_config(dataset_config)

# %%
# Initialize overall statistics
overall_detected = 0
overall_manual = 0
overall_missed = 0

for position in dataset_config.zarr_positions:
    # Combine manual_tp_annotations for the current position
    manual_scope_error = set(
        tp_annotations.get(TimepointAnnotation.BF_SCOPE_ERROR, {}).get(position, [])
    )
    manual_temp_artifact = set(
        tp_annotations.get(TimepointAnnotation.BF_TEMP_ARTIFACT, {}).get(position, [])
    )
    manual_tps = manual_scope_error | manual_temp_artifact

    # Combine auto_tp_annotations for the current position
    auto_tps = set(auto_tp_annotations[TimepointAnnotation.BF_SCOPE_ERROR].get(position, [])) | set(
        auto_tp_annotations.get(TimepointAnnotation.BF_TEMP_ARTIFACT, {}).get(position, [])
    )

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

# %% 0.004 Stats
import numpy as np

total_detected = np.sum([70, 70, 13, 15, 140, 57, 94, 41, 78, 94, 70, 41, 84])
total_manual = np.sum([19, 43, 6, 6, 10, 6, 66, 0, 37, 35, 1, 22, 28])
total_missed = np.sum([1, 5, 0, 3, 1, 2, 4, 0, 0, 0, 0, 0, 0])

percent_missed = (total_missed / total_manual) * 100
print(f"Overall: Detected {total_detected}, Manual {total_manual - 1}, Missed {total_missed}")
print(f"Percent Missed: {percent_missed:.2f}%")

# %%
