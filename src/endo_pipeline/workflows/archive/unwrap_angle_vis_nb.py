# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endo_pipeline.cli.logs import silence_external_loggers
from endo_pipeline.configs import TimepointAnnotation, get_datasets_in_collection
from endo_pipeline.configs.dataset_config_utils import get_subset_of_timepoint_annotations
from endo_pipeline.io import get_output_path, save_plot_to_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import get_pc_column_names
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.library.visualize.multi_feature_correlation_viz import (
    get_df_for_feature_correlation_viz,
)
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.diffae_feature_dataframes import NUM_PCS_TO_ANALYZE
from endo_pipeline.settings.workflow_defaults import (
    DATASET_INFO_COLUMNS,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    SEGMENTATION_FEATURE_COLUMNS,
)

# %%
silence_external_loggers()

fig_savedir = get_output_path(__file__)

ANGLE_COLUMN_NAMES = ["Orientation (deg)", f"{Column.DiffAEData.POLAR_ANGLE}"]
ANGLE_PERIOD = np.pi

# %%

# include "non-steady state" timepoints, filter out the rest
timepoint_annotations = get_subset_of_timepoint_annotations(
    annotations_to_ignore=[TimepointAnnotation.NOT_STEADY_STATE]
)

# default dataset list to get features from (PCA reference datasets)
dataset_name_list = get_datasets_in_collection(DEFAULT_PCA_DATASET_COLLECTION_NAME)

# Long operation: takes several minutes
df = get_df_for_feature_correlation_viz(
    dataset_name_list=dataset_name_list,
    dataset_info_columns=DATASET_INFO_COLUMNS,
    segmentation_feature_columns=SEGMENTATION_FEATURE_COLUMNS["default"],
    pc_columns=get_pc_column_names(NUM_PCS_TO_ANALYZE),
    timepoint_annotations=timepoint_annotations,
)
# %%
# get dataframe of angular columns: orientation of cell
# and the PC-derived polar angle
angle_column_names = [get_label_for_column(name) for name in ANGLE_COLUMN_NAMES]
df_angles = df[angle_column_names].copy()

# convert degrees to radians
df_angles["Orientation (rad)"] = df_angles["Orientation (deg)"].apply(np.deg2rad)
angle_column_names = ["Orientation (rad)"] + angle_column_names
angle_column_names.remove("Orientation (deg)")
df_angles = df_angles[angle_column_names]

# %%
# apply np unwrap to each row, i.e., to each
# pair (orientation, polar angle)
angles_unwrapped = np.unwrap(df_angles[angle_column_names].to_numpy(), period=ANGLE_PERIOD, axis=1)
# create new dataframe with unwrapped angles
angle_column_names_unwrapped = [name + " (unwrapped)" for name in angle_column_names]
df_angles_unwrapped = pd.DataFrame(data=angles_unwrapped, columns=angle_column_names_unwrapped)

# %%
# plot wrapped angles and compute correlation
correlation = df_angles.corr().iloc[0, 1]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    df_angles[angle_column_names[0]],
    df_angles[angle_column_names[1]],
    color="k",
    alpha=0.1,
)
ax.set_xlabel(angle_column_names[0])
ax.set_ylabel(angle_column_names[1])
ax.set_title(f"Wrapped angles (correlation={correlation:.2f})")

save_plot_to_path(
    fig,
    fig_savedir,
    "angle_wrapped_scatter.png",
)

# %%
# plot unwrapped angles and compute correlation
correlation_unwrapped = df_angles_unwrapped.corr().iloc[0, 1]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    df_angles_unwrapped[angle_column_names_unwrapped[0]],
    df_angles_unwrapped[angle_column_names_unwrapped[1]],
    color="k",
    alpha=0.1,
)
ax.set_xlabel(angle_column_names_unwrapped[0])
ax.set_ylabel(angle_column_names_unwrapped[1])
ax.set_title(f"Unwrapped angles (correlation={correlation_unwrapped:.2f})")
save_plot_to_path(
    fig,
    fig_savedir,
    "angle_unwrapped_scatter.png",
)
# %%
