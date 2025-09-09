# %%
import numpy as np
import pandas as pd

from endo_pipeline.configs import (
    TimepointAnnotation,
    get_annotated_timepoints_for_position,
    get_datasets_in_collection,
    load_dataset_config,
    save_dataset_config,
)
from endo_pipeline.io.output import get_output_path
from endo_pipeline.library.process.bf_timepoint_outlier import detect_outliers

# %% UPDATE ANNOTATIONS IN DATASET CONFIGS
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
# for dataset_name in [datasets[0]]: # For testing with a single dataset
for dataset_name in datasets:
    dataset_config = load_dataset_config(dataset_name)
    tp_annotations = (
        dataset_config.timepoint_annotations
        if dataset_config.timepoint_annotations is not None
        else {}
    )

    tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR] = {
        position: [] for position in dataset_config.zarr_positions
    }
    tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT] = {
        position: [] for position in dataset_config.zarr_positions
    }

    for position in dataset_config.zarr_positions:
        bf_scope_error, bf_temp_artifact = detect_outliers(dataset_config, position, visualize=True)

        tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR][position].extend(bf_scope_error)
        tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT][position].extend(bf_temp_artifact)

    dataset_config.timepoint_annotations = tp_annotations
    save_dataset_config(dataset_config)


# %% CALCULATE STATISTICS
stats = []
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
for dataset_name in datasets:
    dataset_config = load_dataset_config(dataset_name)
    for position in dataset_config.zarr_positions:
        manual_annotations = [
            TimepointAnnotation.BF_SCOPE_ERROR,
            TimepointAnnotation.BF_TEMP_ARTIFACT,
        ]
        manual_tps = set(
            get_annotated_timepoints_for_position(dataset_config, position, manual_annotations)
        )

        auto_annotations = [
            TimepointAnnotation.AUTO_BF_SCOPE_ERROR,
            TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT,
        ]
        auto_tps = set(
            get_annotated_timepoints_for_position(dataset_config, position, auto_annotations)
        )

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
# %%
df = pd.DataFrame(stats)
save_dir = get_output_path("brightfield_outlier_detection")
df.to_parquet(save_dir / "bf_outlier_detection_stats.parquet", index=False)

total_manual = df["n_manual_annotated"].sum()
total_auto = df["n_auto_detected"].sum()
total_missed = df["n_missed"].sum() - 1  # -1 b/c one annotation is not expected to be detected
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
