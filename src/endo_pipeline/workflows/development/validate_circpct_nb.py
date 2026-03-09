# %%
import logging

import numpy as np
from scipy.stats import circmean

from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.io import get_output_path
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
    rescale_polar_angle,
    rewrap_polar_angle,
    unrescale_polar_angle,
)
from endo_pipeline.library.analyze.numerics.binning import circpercentile
from endo_pipeline.library.visualize.diffae_features.feature_viz import get_label_for_column
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName
from endo_pipeline.settings.dynamics_workflows import (
    BIN_LIMITS_DYNAMICS,
    BIN_LIMITS_THETA_RESCALED,
    BIN_WIDTHS_DYNAMICS,
    NUM_PCS_TO_FIT_FOR_DYNAMICS,
    RESCALE_THETA,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

logger = logging.getLogger(__name__)
apply_workflow_options(WorkflowOptions(verbose=True))
# %%
model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
run_name = DEFAULT_MODEL_RUN_NAME
crop_pattern = "grid"

dataset_name = "20250409_20X"

# get labels for provided set of feature columns
column_names = [ColumnName.POLAR_ANGLE.value]
variable_labels_dict = {
    col: get_label_for_column(col).replace("polar ", "") for col in column_names
}

# unpack default bin widths and limits for each column, adjusting limits if rescaling theta
global_bin_limits_dict = BIN_LIMITS_DYNAMICS.copy()
if RESCALE_THETA:
    global_bin_limits_dict[ColumnName.POLAR_ANGLE.value] = BIN_LIMITS_THETA_RESCALED
polar_angle_period = (
    global_bin_limits_dict[ColumnName.POLAR_ANGLE.value][1]
    - global_bin_limits_dict[ColumnName.POLAR_ANGLE.value][0]
)
original_polar_range = global_bin_limits_dict[ColumnName.POLAR_ANGLE.value]
bin_widths = [BIN_WIDTHS_DYNAMICS[col] for col in column_names]

# get dataframe manifest for grid-based crop features
model_manifest = load_model_manifest(model_manifest_name)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern=crop_pattern
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

# fit PCA - ALWAYS on grid-based crop features
dataframe_manifest_name_for_pca = get_feature_dataframe_manifest_name(
    model_manifest, run_name, crop_pattern="grid"
)
pca = fit_pca(
    dataframe_manifest_name=dataframe_manifest_name_for_pca, num_pcs=NUM_PCS_TO_FIT_FOR_DYNAMICS
)


# loop over datasets in collection, compute 2D drift coefficients for each
# pairwise combination of polar coordinates, and plot contours of drift coefficients
fig_savedir = get_output_path(__file__, crop_pattern, dataset_name)
logger.debug("Saving summary plots to [ %s ]", fig_savedir)
dataset_config = load_dataset_config(dataset_name)
global_bin_limits_dict[ColumnName.POLAR_ANGLE.value] = original_polar_range

# %%
df = get_dataframe_for_dynamics_workflows(
    dataset_name,
    dataframe_manifest,
    pca=pca,
    include_cell_piling=False,
    include_not_steady_state=False,
    crop_pattern=crop_pattern,
    compute_polar=True,
    rescale_theta=RESCALE_THETA,
)
# %%
angle_values = df[ColumnName.POLAR_ANGLE.value].to_numpy()
unrescaled_angles = unrescale_polar_angle(angle_values)
circ_mean = circmean(unrescaled_angles, low=-np.pi, high=np.pi)
logger.debug("Circular mean of unrescaled angles: [ %s ] radians", circ_mean)


lower_bound = circpercentile(unrescaled_angles, 5)
upper_bound = circpercentile(unrescaled_angles, 95)
logger.debug("5th percentile of unrescaled angles: [ %s ] radians", lower_bound)
logger.debug("95th percentile of unrescaled angles: [ %s ] radians", upper_bound)

lb_rewrapped = rewrap_polar_angle(lower_bound, (-np.pi, np.pi))
ub_rewrapped = rewrap_polar_angle(upper_bound, (-np.pi, np.pi))
logger.debug("Wrapped 5th percentile of unrescaled angles: [ %s ] radians", lb_rewrapped)
logger.debug("Wrapped 95th percentile of unrescaled angles: [ %s ] radians", ub_rewrapped)

lb_rescaled = rescale_polar_angle(lb_rewrapped)
ub_rescaled = rescale_polar_angle(ub_rewrapped)
logger.debug("Rescaled 5th percentile: [ %s ] radians", lb_rescaled)
logger.debug("Rescaled 95th percentile: [ %s ] radians", ub_rescaled)
