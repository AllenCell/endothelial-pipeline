from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from cellsmap.util.set_output import get_output_path
from src.endo_pipeline.configs import get_model_manifest, load_dataset_config, load_model_config
from src.endo_pipeline.configs.dataset_io import (
    get_live_segmentation_features_manifest,
    get_reference_datasets,
    ipython_cli_flexecute,
)
from src.endo_pipeline.configs.dynamics_io import load_dynamics_config
from src.endo_pipeline.io import load_dataframe_from_fms
from src.endo_pipeline.library.analyze.diffae_features import regression_helper as rh
from src.endo_pipeline.library.analyze.diffae_manifest import preprocessing as diffae_preproc
from src.endo_pipeline.library.analyze.diffae_manifest.manifest_pca import fit_pca
from src.endo_pipeline.library.analyze.diffae_manifest.preprocessing import (
    get_manifest_for_dynamics_workflows,
    project_manifest_to_pcs,
)
from src.endo_pipeline.library.analyze.numerics import data_driven_flow_field as ddff
from src.endo_pipeline.library.visualize.diffae_features.flow_field_viz import (
    get_slice_indexes,
    plot_quiver_slices,
    set_slice_plot_bounds_and_labels,
)
from src.endo_pipeline.workflows.feats_diffae_classic_comparison import (  # get_traj_and_flowfield_from_manifest,
    get_traj_and_flowfield,
)

dataset_name = "20241016_20X"
dataset_config = load_dataset_config(dataset_name)

# load the tables
merged_table = load_dataframe_from_fms(
    dataset_config.live_merged_seg_features_manifest_fmsid
)  # this takes a couple of minutes
# diffae_table = load_dataframe_from_fms(dataset_config.diffae_tracking_integration_fmsid)
diffae_table = pd.read_parquet(
    r"C:\Users\serge.parent\OneDrive - Allen Institute\Desktop\temp_look_at\20241016_20X\predict_20241016_20X_diffae_04_10_track_based_features.parquet"
)

# filter the merged table
merged_table = merged_table[~merged_table["filter_global"]]


# fit the PCA (uses the reference datasets)
pca = fit_pca()

# read in the grid crop-based diffae features
model_name = diffae_table["model_name"].unique()[0]
model_config = load_model_config(model_name)
get_model_manifest(dataset_name, model_config)
diffae_grid_crops = get_manifest_for_dynamics_workflows(dataset_name, pca)

# add the PC columns to the track-based DiffAE table
# (the grid-based DiffAE table already has them, but
# but I believe that the columns are named "feat_0",
# "feat_1", etc. when they should be named "pc1",
# "pc2", etc.)
df_all_positions = project_manifest_to_pcs(merged_table, pca)
df_all_positions.dropna(axis="index", how="any", subset="is_unique", inplace=True)

# use the full set of datasets to be analyzed for the bounds
bounds = ddff.set_3d_bounds_from_data([dataset_name], pca)

print("getting trajectory and flow field for grid-based crops...")
traj_grids, flow_field_dict_grids = get_traj_and_flowfield(diffae_grid_crops, bounds)

print("getting trajectory and flow field for tracks-based crops...")
traj_tracks, _ = get_traj_and_flowfield(df_all_positions, bounds)
# save the trajectory data from the track-based crops
np.save(out_subdir_traj / f"{dataset_name}_traj_tracks.npy", traj_tracks)
