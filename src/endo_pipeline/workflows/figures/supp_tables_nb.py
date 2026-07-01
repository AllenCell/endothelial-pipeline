# %%
from matplotlib import pyplot as plt

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.supp_tables import create_supp_table
from endo_pipeline.manifests import load_dataframe_manifest
from endo_pipeline.settings.supp_tables import (
    TABLE_S1_SHEAR_STRESS,
    TABLE_S2_DIFFAE,
    TABLE_S3_PERTURBATION,
    TABLE_S4_NUCLEAR_LABELFREE,
    TABLE_S5_SEGMENTATION,
)
from endo_pipeline.settings.workflow_defaults import (
    CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME,
    FEATURES_FILTERED_MANIFEST_NAMES,
)

plt.style.use("endo_pipeline.figure")

save_dir = get_output_path("supp-fig-tables")
cell_manifest = load_dataframe_manifest(FEATURES_FILTERED_MANIFEST_NAMES["cell_centered"])
seg_manifest = load_dataframe_manifest(CELL_CENTERED_FEATURES_UNFILTERED_MANIFEST_NAME)
shear_stress_datasets = set(get_datasets_in_collection("shear_stress"))
diffae_datasets = set(get_datasets_in_collection("diffae_model_training"))

# %%
create_supp_table(TABLE_S1_SHEAR_STRESS, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S2_DIFFAE, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S3_PERTURBATION, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(TABLE_S4_NUCLEAR_LABELFREE, cell_manifest=cell_manifest, save_dir=save_dir)

# %%
create_supp_table(
    TABLE_S5_SEGMENTATION,
    cell_manifest=seg_manifest,
    save_dir=save_dir,
    shear_stress_datasets=shear_stress_datasets,
    diffae_datasets=diffae_datasets,
)

# %%
