# %%
from endo_pipeline.io import load_dataframe
from endo_pipeline.library.analyze.diffae_dataframe import remove_annotated_timepoints_and_positions
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings import POSITION_COLUMN_NAME, TIMEPOINT_COLUMN_NAME

# %%
model_manifest_name = "diffae_baseline_include_cell_piling"
model_manifest = load_model_manifest(model_manifest_name)
run_name = "20250918_no_log_norm"
dataframe_manifest_name = get_feature_dataframe_manifest_name(model_manifest, run_name)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)

dataset_name = "20250618_20X"
dataset_location = get_dataframe_location_for_dataset(dataframe_manifest, dataset_name)
df = load_dataframe(dataset_location)

# %%
for position, df_pos in df.groupby(POSITION_COLUMN_NAME):
    timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
    timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
    print(
        f"- Position [ {position} ] has timepoints from [ {timepoint_min} ] to [ {timepoint_max} ]"
    )

# %%
df_rm_cell_piling = remove_annotated_timepoints_and_positions(df, remove_not_steady_state=False)
for position, df_pos in df_rm_cell_piling.groupby(POSITION_COLUMN_NAME):
    timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
    timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
    print(
        f"- Position [ {position} ] has timepoints from [ {timepoint_min} ] to [ {timepoint_max} ]"
    )

# %%
df_rm_not_steady_state = remove_annotated_timepoints_and_positions(df, remove_cell_piling=False)
for position, df_pos in df_rm_not_steady_state.groupby(POSITION_COLUMN_NAME):
    timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
    timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
    print(
        f"- Position [ {position} ] has timepoints from [ {timepoint_min} ] to [ {timepoint_max} ]"
    )

# %%
df_rm_both = remove_annotated_timepoints_and_positions(df)
for position, df_pos in df_rm_both.groupby(POSITION_COLUMN_NAME):
    timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
    timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
    print(
        f"- Position [ {position} ] has timepoints from [ {timepoint_min} ] to [ {timepoint_max} ]"
    )

# %%
df_rm_neither = remove_annotated_timepoints_and_positions(
    df, remove_cell_piling=False, remove_not_steady_state=False
)
for position, df_pos in df_rm_neither.groupby(POSITION_COLUMN_NAME):
    timepoint_min = df_pos[TIMEPOINT_COLUMN_NAME].min()
    timepoint_max = df_pos[TIMEPOINT_COLUMN_NAME].max()
    print(
        f"- Position [ {position} ] has timepoints from [ {timepoint_min} ] to [ {timepoint_max} ]"
    )
