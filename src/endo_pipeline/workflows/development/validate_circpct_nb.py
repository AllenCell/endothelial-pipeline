# %%
import logging

from scipy.stats import circmean

from endo_pipeline.cli.apps import WorkflowOptions, apply_workflow_options
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
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

dataset_name = "20250618_20X"

# %%
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
circ_mean = circmean(
    angle_values, low=BIN_LIMITS_THETA_RESCALED[0], high=BIN_LIMITS_THETA_RESCALED[1]
)
logger.info("Circular mean of angles: [ %s ] radians", circ_mean)

lower_bound = circpercentile(angle_values, 5, polar_range=BIN_LIMITS_THETA_RESCALED)
upper_bound = circpercentile(angle_values, 95, polar_range=BIN_LIMITS_THETA_RESCALED)
logger.info("5th percentile of angles: [ %s ] radians", lower_bound)
logger.info("95th percentile of angles: [ %s ] radians", upper_bound)
# %%
