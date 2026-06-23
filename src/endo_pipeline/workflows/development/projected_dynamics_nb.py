# %%
from endo_pipeline.io import get_output_path
from endo_pipeline.library.visualize.diffae_features.projected_dynamics import (
    visualize_projected_dynamics,
)

# %%
output_path = get_output_path("projected_dynamics")

# %%
for dataset_name in [
    "20250319_20X",
    "20260216_20X",
    "20250813_20X",
    "20260114_20X",
    "20260211_20X",
]:
    save_path = visualize_projected_dynamics(dataset_name=dataset_name, output_path=output_path)
    print(f"Saved projected dynamics figure for dataset {dataset_name} to {save_path}.")

# %%
