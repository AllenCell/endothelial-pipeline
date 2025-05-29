# %%
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cellsmap.analyses.immunofluorescence.if_support.add_if_cols import (
    get_channels_for_if_processing,
)
from cellsmap.analyses.integration.feats_diffae_classic_comparison import (
    get_traj_and_flowfield,
    plot_measured_feat_overlay_on_flowfield,
    plot_quiver_slices_from_diffae_table,
)
from cellsmap.analyses.utils.numerics import data_driven_flow_field as ddff
from cellsmap.util.dataset_io import get_reference_datasets
from cellsmap.util.manifest_preprocessing.diffae_feature_preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from cellsmap.util.manifest_preprocessing.manifest_pca import fit_pca
from cellsmap.util.set_output import get_output_path

# %%
# load output from preprocessing step
# dataframe of all if datasets
df_if = pd.load_csv(
    "immunoflourescence_analysis_integration/outputs/immunofluorescence_manifest.csv"
)
# %% Calculate PCA and bounds for the reference dataset
pca = fit_pca()
reference_datasets = get_reference_datasets()
bounds = ddff.set_3d_bounds_from_data(reference_datasets, pca, col_names="feat")
# %%
conditions = [
    {
        "ref_dataset_name": "20241120_20X",  # 48 hour high flow
        "flow_regime": "high_flow",
        "corresponding_if_datasets": [
            "20250509_20X_IF1",
            "20250509_20X_IF9",
            "20250509_20X_IF10",
        ],
    },
    {
        "ref_dataset_name": "20250409_20X",  # low flow
        "flow_regime": "low_flow",
        "corresponding_if_datasets": [
            "20250509_20X_IF5",
            "20250509_20X_IF6",
            "20250509_20X_IF12",
        ],
    },
    {
        "ref_dataset_name": "20241217_20X",  # no flow
        "flow_regime": "no_flow",
        "corresponding_if_datasets": [
            "20250509_20X_IF2",
            "20250509_20X_IF3",
            "20250509_20X_IF4",
        ],
    },
]
# %%
for condition in conditions:
    dataset_name = str(condition["ref_dataset_name"])
    flow_regime = condition["flow_regime"]
    corresponding_if_datasets = condition["corresponding_if_datasets"]

    print(f"Processing dataset: {dataset_name}, Flow regime: {flow_regime}")

    output_dir = get_output_path(
        f"immunoflourescence_analysis_integration/{dataset_name}_{flow_regime}"
    )
    diffae_grid_crops = get_manifest_for_dynamics_workflows(dataset_name, pca)
    print("getting trajectory and flow field for grid-based crops...")
    traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
        diffae_grid_crops, bounds, col_names="feat"
    )

    for if_dataset_name in corresponding_if_datasets:

        df_condition = df_if[df_if["dataset"].isin(corresponding_if_datasets)]

        df_if_dataset = df_if[df_if["dataset"] == if_dataset_name]

        df_if_dataset = project_manifest_to_pcs(
            df_if_dataset, pca, overwrite_feature_columns=False
        )

        # this is where we would apply a PC correction to PC1, 2 and 3

        channels = get_channels_for_if_processing(if_dataset_name)
        print(f"Plotting {if_dataset_name}")

        for channel in channels:
            if channel == "NucViolet":
                continue

            feature = f"crop_nuc_mean_intensity_{channel}"
            hue_min = df_condition[feature].min()
            hue_max = df_condition[feature].max()

            plot_measured_feat_overlay_on_flowfield(
                Path(output_dir),
                if_dataset_name,
                diffae_grid_crops,
                traj_grids,
                flow_field_dict_grids,
                diffae_measured_feat_df=df_if_dataset,
                meas_feat_col_name_for_color_coding=feature,
                track_id_to_plot=None,
                show_plot=True,
                alpha=1.0,
                hue_norm=None,
            )

# %%
# generate plots for the SAC-2025 slides
# NOTE I (SERGE) WROTE THIS SHORTLY BEFORE SAC SLIDES WERE DUE;
# IT NEEDS TO BE REFACTORED AND BETTER INTEGRATED WITH OTHER CODE
for condition in conditions:
    if "20250509_20X_IF9" in condition["corresponding_if_datasets"]:
        break

dataset_name = str(condition["ref_dataset_name"])
flow_regime = condition["flow_regime"]
corresponding_if_datasets = condition["corresponding_if_datasets"]

print(f"Processing dataset: {dataset_name}, Flow regime: {flow_regime}")

output_dir = get_output_path(f"immunoflourescence_analysis_integration/SAC_2025")
diffae_grid_crops = get_manifest_for_dynamics_workflows(dataset_name, pca)
print("getting trajectory and flow field for grid-based crops...")
traj_grids, flow_field_dict_grids = get_traj_and_flowfield(
    diffae_grid_crops, bounds, col_names="feat"
)

if_dataset_name = "20250509_20X_IF9"  # SAC-2025 slide dataset

df_condition = df_if[df_if["dataset"].isin(corresponding_if_datasets)]

df_if_dataset = df_if[df_if["dataset"] == if_dataset_name]

df_if_dataset = project_manifest_to_pcs(
    df_if_dataset, pca, overwrite_feature_columns=False
)

# this is where we would apply a PC correction to PC1, 2 and 3

channels = get_channels_for_if_processing(if_dataset_name)
print(f"Plotting {if_dataset_name}")

channel = "SMAD1"

feature = f"crop_nuc_mean_intensity_{channel}"
hue_min = df_condition[feature].min()
hue_max = df_condition[feature].max()

# plot and save just the flow field first
Path(output_dir).mkdir(exist_ok=True, parents=True)

fig, axs = plot_quiver_slices_from_diffae_table(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)
plt.tight_layout()
fig.savefig(
    Path(output_dir) / f"{dataset_name}_flow_field_quiver.png",
    dpi=300,
)
plt.close(fig)

# next plot and save the flow field with the measured feature overlay
plot_measured_feat_overlay_on_flowfield(
    Path(output_dir),
    if_dataset_name,
    diffae_grid_crops,
    traj_grids,
    flow_field_dict_grids,
    diffae_measured_feat_df=df_if_dataset,
    meas_feat_col_name_for_color_coding=feature,
    track_id_to_plot=None,
    show_plot=True,
    alpha=1.0,
    hue_norm=None,
)
plt.close(fig)

# lastly save the quiver slices with just
# the diffae scatter points overlayed
# this is the same as the above, but without the measured feature overlay

fig, axs = plot_quiver_slices_from_diffae_table(
    diffae_grid_crops, traj_grids, flow_field_dict_grids
)

pc_x, pc_y = ("pc1", "pc1"), ("pc2", "pc3")

for i, ax in enumerate(axs):
    sns.scatterplot(
        data=df_if_dataset,
        x=pc_x[i],
        y=pc_y[i],
        color="navy",
        linewidth=0,
        marker=".",
        s=75,
        alpha=1.0,
        ax=ax,
    )
plt.tight_layout()
fig.savefig(
    Path(output_dir) / f"{dataset_name}_flow_field_{channel}.png",
    dpi=300,
)
plt.close(fig)


# %%
