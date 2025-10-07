# %%
from endo_pipeline.configs import (
    TimepointAnnotation,
    get_datasets_in_collection,
    load_dataset_config,
    save_dataset_config,
)
from endo_pipeline.library.process.bf_timepoint_outlier import detect_bf_outliers
from endo_pipeline.library.process.gfp_timepoint_outlier import detect_egfp_scope_errors

# %% UPDATE ANNOTATIONS IN DATASET CONFIGS
datasets = get_datasets_in_collection("live_20X_objective_3i_microscope")
# datasets = [datasets[0]] # For testing with a single dataset
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
    tp_annotations[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR] = {
        position: [] for position in dataset_config.zarr_positions
    }

    for position in dataset_config.zarr_positions:
        bf_scope_error, bf_temp_artifact = detect_bf_outliers(
            dataset_config, position, visualize=True
        )
        egfp_scope_error = detect_egfp_scope_errors(dataset_config, position, visualize=True)

        tp_annotations[TimepointAnnotation.AUTO_BF_SCOPE_ERROR][position].extend(bf_scope_error)
        tp_annotations[TimepointAnnotation.AUTO_BF_TEMP_ARTIFACT][position].extend(bf_temp_artifact)
        tp_annotations[TimepointAnnotation.AUTO_GFP_SCOPE_ERROR][position].extend(egfp_scope_error)

    dataset_config.timepoint_annotations = tp_annotations
    save_dataset_config(dataset_config)
