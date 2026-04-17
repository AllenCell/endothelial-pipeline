# %%
from endo_pipeline.io import get_output_path, load_dataframe, load_model, save_plot_to_path
from endo_pipeline.library.model.diffae.generate_image import generate_from_dataframe
from endo_pipeline.library.visualize.figure_utils import make_contact_sheet
from endo_pipeline.manifests import load_dataframe_manifest, load_model_manifest
from endo_pipeline.settings.dynamics_workflows import DYNAMICS_COLUMN_NAMES
from endo_pipeline.settings.flow_field_dataframes import DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS
from endo_pipeline.settings.migration_coherence import MIGRATION_COHERENCE_CROP_PATTERN
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    RANDOM_SEED,
)

# %%
save_dir = get_output_path("generate_from_dataframe")

# Load fixed point dataframe
dataset_name = "20250409_20X"
base_name = (
    f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_{MIGRATION_COHERENCE_CROP_PATTERN}"
)
fixed_points_dataframe_manifest_name = f"{DATAFRAME_MANIFEST_PREFIX_FIXED_POINTS}_{base_name}"
fixed_points_dataframe_manifest = load_dataframe_manifest(fixed_points_dataframe_manifest_name)
fixed_points_dataframe = load_dataframe(fixed_points_dataframe_manifest.locations[dataset_name])

# load model for image generation
model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)

# %%
n_noise_samples = 4
generated_images = generate_from_dataframe(
    fixed_points_dataframe,
    column_names=list(DYNAMICS_COLUMN_NAMES),
    model=model,
    random_seed=RANDOM_SEED,
    n_noise_samples=n_noise_samples,
)
# %%
image_list = [generated_images[i] for i in range(generated_images.shape[0])]
fig, ax = make_contact_sheet(panels=image_list, max_rows=2, max_cols=2)
save_plot_to_path(fig, save_dir, "generated_image")
