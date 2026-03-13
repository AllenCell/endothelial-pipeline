from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import (
    create_dataframe_manifest,
    get_dataframe_location_for_dataset,
    list_datasets_with_dataframes,
    load_dataframe_manifest,
    load_model_manifest,
)

# Define manifest names
model_manifest_name = "diffae_baseline_exclude_cell_piling"
run_name = "20251110_latent_512"
crop_pattern = "grid"
dataframe_manifest_name = f"{model_manifest_name}_{run_name}_{crop_pattern}"
feature_manifest_name = f"{model_manifest_name}_{run_name}_{crop_pattern}_position_as_string"

# Load model and dataframe manifests
model_manifest = load_model_manifest(model_manifest_name)
manifest = load_dataframe_manifest(dataframe_manifest_name)
feature_manifest = create_dataframe_manifest(feature_manifest_name)

# Create temporary output directory
output_dir = get_output_path("temp_position_conversion", include_timestamp=False)

for dataset in list_datasets_with_dataframes(manifest):
    print(dataset)

    # Load existing dataframe for current dataset
    location = get_dataframe_location_for_dataset(manifest, dataset)
    df = load_dataframe(location)

    # Swap integer positions to string positions
    position_col_as_str = df.position
    position_col_as_int = position_col_as_str.apply(lambda x: int(x.replace("P", "")))
    df["position"] = position_col_as_int

    # Save to local output location
    filename_suffix = f"{dataset}_{dataframe_manifest_name}_features"
    output_path = output_dir / f"predict_{filename_suffix}.parquet"
    df.to_parquet(output_path)
