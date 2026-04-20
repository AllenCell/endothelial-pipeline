# %%
import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe, load_image, save_plot_to_path
from endo_pipeline.library.analyze.dataframe_filtering import filter_dataframe_to_binned_value
from endo_pipeline.library.analyze.numerics.binning import get_bins
from endo_pipeline.library.process.image_processing import max_proj, std_dev
from endo_pipeline.library.visualize.figure_utils import add_scalebar, make_contact_sheet
from endo_pipeline.manifests import get_zarr_location_for_position, load_dataframe_manifest
from endo_pipeline.settings.column_names import ColumnName as Column
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_DYNAMICS,
    BIN_LIMITS_THETA_RESCALED,
    DYNAMICS_COLUMN_NAMES,
)
from endo_pipeline.settings.figures import FONTSIZE_SMALL, MAX_FIGURE_WIDTH
from endo_pipeline.settings.image_data import PIXEL_SIZE_3i_20x
from endo_pipeline.settings.plot_defaults import CROP_HIST_BIN_WIDTH
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    RANDOM_SEED,
)

# %%
logger = logging.getLogger(__name__)

# Default list of datasets if not provided. Otherwise, use the provided list.
dataset_name = "20250409_20X"

# get dataframe manifest corresponding to the model that generated the features
# get dataframe manifest for crop-based features
base_name = f"{DEFAULT_MODEL_MANIFEST_NAME}_{DEFAULT_MODEL_RUN_NAME}_grid"
feature_dataframe_manifest_name = f"{base_name}_pca_filtered"
feature_dataframe_manifest = load_dataframe_manifest(feature_dataframe_manifest_name)

fig_savedir = get_output_path(__file__)

dataframe_location = feature_dataframe_manifest.locations[dataset_name]
df = load_dataframe(dataframe_location)

feat_cols = list(DYNAMICS_COLUMN_NAMES)

bin_limits = []
for col in feat_cols:
    if col == Column.DiffAEData.POLAR_ANGLE:
        col_min, col_max = BIN_LIMITS_THETA_RESCALED
    else:
        col_min, col_max = BIN_LIMITS_DYNAMICS.get(col, (df[col].min(), df[col].max()))

    bin_limits.append((col_min, col_max))

# example point in (theta, r, rho) space to bin around to find nearby examples
# images in the dataset
EXAMPLE_POINT = np.array([2.9, 1.0, 0.0])
bin_widths = [CROP_HIST_BIN_WIDTH] * len(feat_cols)
bin_edges = get_bins(bin_widths=bin_widths, bin_limits=bin_limits)[0]

# %%
df = filter_dataframe_to_binned_value(df, feat_cols, EXAMPLE_POINT, bin_edges)

# one example selected at random
num_crop_samples = 4
df_sample = df.sample(n=num_crop_samples, random_state=RANDOM_SEED, replace=False)

print(
    f"Selected crops near point: ({', '.join(feat_cols)}) = ({', '.join(map(str, EXAMPLE_POINT))})"
)
print("Range of values in each feature column for selected crops:")
for col in feat_cols:
    col_min = df_sample[col].min()
    col_max = df_sample[col].max()
    print(f"    {col}: [{col_min:.3f}, {col_max:.3f}]")
# %%
crops_bf_std_deviation = []
crops_gfp_max_projection = []

dataset_config = load_dataset_config(dataset_name)
for sample_idx in range(num_crop_samples):
    df_crop = df_sample.iloc[sample_idx : sample_idx + 1]
    position = df_crop[Column.POSITION].iloc[0]
    timepoint = df_crop[Column.TIMEPOINT].iloc[0]

    img_loc = get_zarr_location_for_position(dataset_config, position)
    img = load_image(img_loc, timepoints=[timepoint], level=1, squeeze=True)
    # crop
    start_x = df_crop[f"{Column.DiffAEData.START_X}"].iloc[0]
    start_y = df_crop[f"{Column.DiffAEData.START_Y}"].iloc[0]
    crop_size_x = df_crop[f"{Column.DiffAEData.CROP_SIZE_X}"].iloc[0]
    crop_size_y = df_crop[f"{Column.DiffAEData.CROP_SIZE_Y}"].iloc[0]

    crop = img[:, :, start_y : start_y + crop_size_y, start_x : start_x + crop_size_x]

    # Extract channels once, these channel indices are hardcoded
    # because we defined the order of channels in the zarr
    bf_channel = crop[1, :, :, :].squeeze()
    gfp_channel = crop[0, :, :, :].squeeze()

    # Process channels
    std_dev_proj = std_dev(bf_channel, 0)
    log_norm_std = np.log1p(std_dev_proj)
    low, high = np.percentile(log_norm_std, [0.1, 99.9])
    clipped_std = np.clip(log_norm_std, low, high)
    crops_bf_std_deviation.append(clipped_std)

    cdh5_max_proj = max_proj(gfp_channel, 0)
    low, high = np.percentile(cdh5_max_proj, [10, 98])
    clipped_cdh5 = np.clip(cdh5_max_proj, low, high)
    crops_gfp_max_projection.append(clipped_cdh5)

# %%
# Create panels doing the len of pc_val_list every other contrasted crop type
panels = []
for i in range(num_crop_samples):
    panels.append(crops_bf_std_deviation[i])
for j in range(num_crop_samples):
    panels.append(crops_gfp_max_projection[j])

fig = make_contact_sheet(
    panels,
    max_rows=2,
    max_cols=num_crop_samples,
    fig_kwargs={"layout": "tight", "figsize": (MAX_FIGURE_WIDTH, len(feat_cols) * 2)},
    font_size=FONTSIZE_SMALL,
)
for ax in fig.axes:
    for spine in ax.spines.values():
        spine.set_visible(False)
scale_bar_um = 20
add_scalebar(
    fig.axes[0],
    scale_bar_um=scale_bar_um,
    pixel_size=PIXEL_SIZE_3i_20x,
    bar_thickness=10,
    padding=10,
)

plt.show()
save_plot_to_path(
    fig,
    fig_savedir,
    f"get_real_crop_{'_'.join(map(str, feat_cols))}_{scale_bar_um}um_scalebar",
)

# %%
