# %%
import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    get_datasets_in_collection,
    load_dataset_config,
    save_dataset_config,
)
from endo_pipeline.configs.dataset_config import TimepointAnnotation
from endo_pipeline.io.output import get_output_path
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
print(dataset_config)
# %%
save_dataset_config(dataset_config)

# %% Calculate statistics
stats = []
for dataset_name in datasets:
    dataset_config = load_dataset_config(dataset_name)
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
        auto_scope_error = set(
            auto_tp_annotations.get(TimepointAnnotation.AUTO_BF_SCOPE_ERROR, {}).get(position, [])
        )
        auto_temp_artifact = set(
            auto_tp_annotations.get(TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT, {}).get(position, [])
        )
        auto_tps = auto_scope_error | auto_temp_artifact
        list_of_missed_tps = list(manual_tps - auto_tps) if manual_tps - auto_tps else np.NaN

        stats.append(
            {
                "dataset_name": dataset_name,
                "position": position,
                "n_auto_detected": len(auto_tps),
                "n_manual_annotated": len(manual_tps),
                "n_missed": len(manual_tps - auto_tps),
                "list_of_missed_tps": list_of_missed_tps,
                "n_tps_assessed": dataset_config.duration,
            }
        )

# Save statistics to a DataFrame
df = pd.DataFrame(stats)
save_dir = get_output_path("brightfield_outlier_detection")
df.to_parquet(save_dir / "bf_outlier_detection_stats.parquet", index=False)

# %% Calculate overall stats
total_manual = df["n_manual_annotated"].sum()
total_auto = df["n_auto_detected"].sum()
total_missed = df["n_missed"].sum() - 1
percent_missed = (total_missed / total_manual) * 100 if total_manual > 0 else 0
total_timepoints = df["n_tps_assessed"].sum()
percent_artifact = (total_auto + total_missed) / total_timepoints * 100

print(f"Total manual annotated timepoints: {total_manual}")
print(f"Total missed timepoints: {total_missed}")
print(f"Percent of missed timepoints: {percent_missed:.2f}%")
print(f"Percent of captured timepoints: {100 - percent_missed:.2f}%")
print(f"Total auto-detected timepoints: {total_auto}")
print(f"Total timepoints assessed: {total_timepoints}")
print(f"Percent of tps with artifacts: {percent_artifact:.2f}%")
# %%
