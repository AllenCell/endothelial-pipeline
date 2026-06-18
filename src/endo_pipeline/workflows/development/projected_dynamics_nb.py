# %%
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.visualize.diffae_features.projected_dynamics import (
    visualize_projected_dynamics,
)

# %%
output_path = get_output_path("projected_dynamics")

# %%
for dataset_name in ["20250319_20X", "20250813_20X"]:
    fig = visualize_projected_dynamics(dataset_name=dataset_name)
    _ = save_plot_to_path(fig, output_path, figure_name=f"{dataset_name}_projected_dynamics")

# %%
