from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
from endo_pipeline.manifests import (
    DataframeLocation,
    create_dataframe_manifest,
    list_datasets_with_dataframes,
    load_dataframe_manifest,
    load_model_manifest,
    save_dataframe_manifest,
)

apply_workflow_options(WorkflowOptions(use_staging=False, verbose=True))

# Define manifest names
model_manifest_name = "diffae_baseline_exclude_cell_piling"
run_name = "20251110_latent_512"
crop_pattern = "grid"
dataframe_manifest_name = f"{model_manifest_name}_{run_name}_{crop_pattern}_position_as_str"
feature_manifest_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"

# Load model and dataframe manifests
model_manifest = load_model_manifest(model_manifest_name)
manifest = load_dataframe_manifest(dataframe_manifest_name)
feature_manifest = create_dataframe_manifest(feature_manifest_name)

# Create temporary output directory
output_dir = get_output_path("temp_position_conversion", include_timestamp=False)

for dataset in list_datasets_with_dataframes(manifest):
    print(dataset)

    # Save to local output location
    filename_suffix = f"{dataset}_{feature_manifest_name}_features"
    output_path = output_dir / f"predict_{filename_suffix}.parquet"

    # Upload to FMS
    dataset_config = load_dataset_config(dataset)
    annotations = build_fms_annotations(
        dataset_config, model_manifest=model_manifest, run_name=run_name
    )
    fmsid = upload_file_to_fms(output_path, annotations=annotations, file_type="parquet")
    feature_manifest.locations[dataset] = DataframeLocation(fmsid=fmsid)
    save_dataframe_manifest(feature_manifest)
