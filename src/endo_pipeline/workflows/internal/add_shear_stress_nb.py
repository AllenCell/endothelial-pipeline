# %%

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import (
    build_fms_annotations,
    get_output_path,
    load_dataframe,
    upload_file_to_fms,
)
from endo_pipeline.manifests import DataframeLocation, load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.flow_field_dataframes import FMS_ANNOTATION_NOTES_BOOTSTRAPPING
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

# %%
bootstrap_dataframe_manifest = load_dataframe_manifest(
    "bootstrapped_fixed_points_diffae_baseline_exclude_cell_piling_20251110_latent_512_grid"
)
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)

for dataset_name in bootstrap_dataframe_manifest.locations.keys():
    dataset_config = load_dataset_config(dataset_name)
    dataframe = load_dataframe(bootstrap_dataframe_manifest.locations[dataset_name])
    if Column.SHEAR_STRESS not in dataframe.columns:
        print(
            f"Adding shear stress column to bootstrapped fixed points dataframe for dataset [ {dataset_name} ]."
        )
        if len(dataset_config.flow_conditions) > 1:
            raise ValueError(
                f"Dataset [ {dataset_name} ] has multiple flow conditions, cannot assign single shear stress value."
            )
        else:
            shear_stress = dataset_config.flow_conditions[0].shear_stress
            dataframe[Column.SHEAR_STRESS] = shear_stress

    output_dir = get_output_path("update_bootstrap")
    output_save_path = output_dir / f"{dataset_name}_bootstrapped_fixed_points.parquet"
    dataframe.to_parquet(output_save_path)

    annotations = build_fms_annotations(
        dataset_config,
        model_manifest=model_manifest,
        run_name=DEFAULT_MODEL_RUN_NAME,
        additional_notes=FMS_ANNOTATION_NOTES_BOOTSTRAPPING,
    )
    fmsid = upload_file_to_fms(output_save_path, annotations=annotations, file_type="parquet")
    bootstrap_dataframe_manifest.locations[dataset_name] = DataframeLocation(fmsid=fmsid)

# %%
