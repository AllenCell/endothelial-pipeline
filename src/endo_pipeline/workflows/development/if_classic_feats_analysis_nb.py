# %%
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.library.analyze.immunofluorescence.filter import filter_img_center
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.if_defaults import PLOT_FEAT_COLS, PLOT_FEAT_NAMES
from endo_pipeline.settings.workflow_defaults import FIXED_SEG_FEATURE_MANIFEST_NAME

# %%
output_dir = get_output_path("SMAD1")
smad1_datasets = get_datasets_in_collection("smad1")
if_df_manifest = load_dataframe_manifest("immunofluorescence")
classic_df_manifest = load_dataframe_manifest(FIXED_SEG_FEATURE_MANIFEST_NAME)

dataframe_list = []
for dataset_name in smad1_datasets:
    dataset_config = load_dataset_config(dataset_name)

    df_location_if = get_dataframe_location_for_dataset(if_df_manifest, dataset_name)
    df_dataset_if = load_dataframe(df_location_if)

    # prepare df_dataset_if for merge
    # rename label to nuclei_seg_label
    df_dataset_if = df_dataset_if.rename(columns={"label": "nuclei_seg_label"})
    df_dataset_if.dropna(
        subset=[
            "centroid_X",
            "centroid_Y",
            "image_size_x",
            "image_size_y",
            "zarr_path",
            "image_index",
            "shear_stress_regime",
        ],
        inplace=True,
    )

    df_location_classic = get_dataframe_location_for_dataset(classic_df_manifest, dataset_name)
    df_dataset_classic = load_dataframe(df_location_classic)

    df_merge = pd.merge(
        df_dataset_if,
        df_dataset_classic,
        left_on=[
            "position",
            "centroid_x",
            "centroid_y",
            # "nuclei_seg_label",
        ],
        right_on=[
            "position",
            "nuc_with_most_overlap_0_centroid_X",
            "nuc_with_most_overlap_0_centroid_Y",
            # "nuclei_seg_with_most_overlap_0",
        ],
        how="inner",
        suffixes=("_if", ""),
    )
    num_nans_classic = df_dataset_classic["nuc_with_most_overlap_0_centroid_X"].isna().sum()
    expected_length = len(df_dataset_classic) - num_nans_classic
    if len(df_merge) == expected_length:
        print(f"Dataset {dataset_name} merge complete.")
    else:
        print(
            f"Dataset {dataset_name}, Merge length {len(df_merge)} does not match expected length {expected_length}"
        )

    dataframe_list.append(df_merge)

    # dataframes = [df_dataset_if, df_dataset_classic, df_merge]
    # df_descriptions = ['IF DataFrame', 'Classic DataFrame', 'Merged DataFrame']
    # colors = ['blue', 'green', 'red']
    # markers = ['o', 's', '^']
    # centroid_cols = [
    #     ['centroid_x', 'centroid_y'],
    #     ['nuc_with_most_overlap_0_centroid_X', 'nuc_with_most_overlap_0_centroid_Y'],
    #     ['centroid_x', 'centroid_y']
    # ]

    # # Get all unique positions across all dataframes
    # all_positions = set()
    # for df in dataframes:
    #     all_positions.update(df['position'].unique())

    # for position in sorted(all_positions):
    #     plt.figure()
    #     for df, description, colr, centroid_col, marker in zip(dataframes, df_descriptions, colors, centroid_cols, markers):
    #         df_position = df[df['position'] == position]
    #         if not df_position.empty:
    #             plt.scatter(
    #                 df_position[centroid_col[0]],
    #                 df_position[centroid_col[1]],
    #                 alpha=0.5,
    #                 color=colr,
    #                 marker=marker,
    #                 label=f'{description} - P{position}, N={len(df_position)}'
    #             )
    #     plt.title(f'Position: {position}')
    #     plt.xlabel('centroid_x')
    #     plt.ylabel('centroid_y')
    #     plt.legend(loc="upper right")
    #     plt.show()
# %%
df_all = pd.concat(dataframe_list, ignore_index=True)

# Filter dataframe for analysis:
# hotspot for immunoflourencence intensity comparisons
df = filter_img_center(df_all)
# Reasonable size thresholds for cell and nuclear area
df = df[df["area_if"] >= 450]
df = df[df["area_if"] <= 1600]
df = df[df["area"] >= 1000]
df = df[df["area"] <= 30000]

print(f"Initial dataframe length before filtering: {len(df_all)}")
print(f"Final dataframe length after filtering: {len(df)}")

classic_cols = ["area", "area_if"]
names = ["Cell 2D area (pixels)", "Nuclear 2D area (pixels)"]
for classic_col, name in zip(classic_cols, names, strict=False):
    plt.figure()
    plt.hist(df[classic_col], bins=50, alpha=0.7, label=f"Total N={len(df)}")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.legend()
    plt.show()
# %%
# Build a string-to-color mapping (instead of Enum for quick plotting)
SHEAR_COLOR_STR_DICT = {
    "no": "tab:green",
    "min": "tab:orange",
    "low": "tab:red",
    "medium": "tab:purple",
    "high": "tab:cyan",
    "max": "tab:blue",
    ("min_to_max"): "tab:brown",
    ("max_to_min"): "tab:olive",
}
classic_cols = ["alignment_deg_rel_to_flow", "cell_orientation", "num_nuclei_in_crop"]


# %%
def plot_scatter(df, groupby_cols=None, exclude_no=False, date=None):
    if exclude_no:
        df = df[df["shear_stress_regime"] != "no"]
    if groupby_cols:
        grouped = df.groupby(groupby_cols)
    else:
        grouped = [(None, df)]
    for group_keys, group_df in grouped:
        ss = group_df["shear_stress_regime"].iloc[0] if "shear_stress_regime" in group_df else None
        if "shear_stress_regime" in groupby_cols:
            color = SHEAR_COLOR_STR_DICT.get(ss, "gray")
            label = f"Shear stress: {ss}\nN={len(group_df)}"
        else:
            color = "gray"
            label = f"N={len(group_df)}"
        for if_col, if_feat_name in zip(PLOT_FEAT_COLS, PLOT_FEAT_NAMES, strict=False):
            for classic_col in classic_cols:
                r, p = pearsonr(group_df[if_col], group_df[classic_col])
                p_str = "<0.001" if p < 0.001 else f"{p:.3g}"
                title_date = (
                    date
                    if date
                    else (group_keys[0] if groupby_cols and "date" in groupby_cols else "")
                )
                if p > 0.1:
                    continue
                plt.figure()
                plt.scatter(
                    group_df[if_col], group_df[classic_col], alpha=0.5, color=color, label=label
                )
                plt.xlabel(if_feat_name)
                plt.ylabel(classic_col)
                plt.title(f"{title_date} Pearson r={r:.3f}, p={p_str}")
                plt.legend(loc="upper right")
                plt.show()


plot_scatter(df, groupby_cols=["date", "shear_stress_regime"])
plot_scatter(df, groupby_cols=["shear_stress_regime"])
plot_scatter(df, exclude_no=True)  # all data under shear stress
# %%
