# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.endo_pipeline.configs import get_model_manifest, load_model_config
from src.endo_pipeline.library.analyze.diffae_manifest import (
    fit_pca,
    get_manifest_for_dynamics_workflows,
    get_pc_column_names,
)
from src.endo_pipeline.library.analyze.numerics import get_3d_bounds_from_data
from src.endo_pipeline.library.visualize import viz_base


# %%
def plot_scatter_by_position_and_frame(
    df: pd.DataFrame, target_frame: int
) -> tuple[plt.Figure, np.ndarray]:
    fig, ax = viz_base.init_subplots(figsize=(15, 5))
    pc_column_names = get_pc_column_names(df, [0, 1, 2])

    target_frame = 0

    for position, df_pos in df.groupby("position"):
        df_ = df_pos[df_pos["frame_number"] == target_frame]
        # first plot: PC1 v PC2
        ax[0].scatter(df_[pc_column_names[0]], df_[pc_column_names[1]], s=20)

        # second plot: PC1 v PC3
        ax[1].scatter(df_[pc_column_names[0]], df_[pc_column_names[2]], s=20, label=position)

    ax[0].set_xlim(bounds[0])
    ax[0].set_ylim(bounds[1])
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    ax[1].set_xlim(bounds[0])
    ax[1].set_ylim(bounds[2])
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC3")

    ax[1].legend(loc=(1.05, 0.75))
    fig.suptitle(f"Frame {target_frame}")

    return fig, ax


# %%
model_config = load_model_config("diffae_04_10")
z_stack_offsets = [5, 16]
dataset_name = "20241016_20X"
model_manifest = get_model_manifest(dataset_name, model_config, z_stack_offsets)
# %%
pca = fit_pca()
bounds = get_3d_bounds_from_data([model_manifest], pca)
df = get_manifest_for_dynamics_workflows(model_manifest, pca)

# %%
target_frame = 0
fig, ax = plot_scatter_by_position_and_frame(df, target_frame)
# %%
