# %%

import pandas as pd

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.io import get_output_path, load_dataframe
from endo_pipeline.manifests import get_dataframe_location_for_dataset, load_dataframe_manifest
from endo_pipeline.settings.workflow_defaults import FIXED_SEG_FEATURE_MANIFEST_NAME

# %%
output_dir = get_output_path("SMAD1")
smad1_datasets = get_datasets_in_collection("smad1")
if_df_manifest = load_dataframe_manifest("immunofluorescence")
classic_df_manifest = load_dataframe_manifest(FIXED_SEG_FEATURE_MANIFEST_NAME)

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
            "nuclei_seg_label",
        ],
        right_on=[
            "position",
            "nuc_with_most_overlap_0_centroid_X",
            "nuc_with_most_overlap_0_centroid_Y",
            "nuclei_seg_with_most_overlap_0",
        ],
        how="inner",
        suffixes=("", "_if"),
    )
    num_nans_classic = df_dataset_classic["nuc_with_most_overlap_0_centroid_X"].isna().sum()
    expected_length = len(df_dataset_classic) - num_nans_classic
    if len(df_merge) == expected_length:
        print(f"successfully merged dataset {dataset_name}.")
    else:
        print(
            f"Dataset {dataset_name}, Merge length {len(df_merge)} does not match expected length {expected_length}"
        )

        # dataframes = [df_dataset_if, df_dataset_classic, df_merge]
        # df_descriptions = ['IF DataFrame', 'Classic DataFrame', 'Merged DataFrame']
        # colors = ['blue', 'green', 'red']
        # centroid_cols = [['centroid_x', 'centroid_y'], ['nuc_with_most_overlap_0_centroid_X', 'nuc_with_most_overlap_0_centroid_Y'], ['centroid_x', 'centroid_y']]
        # for df, description, colr, centroid_col in zip(dataframes, df_descriptions, colors, centroid_cols):
        #     for position, df_position in df.groupby('position'):
        #         plt.figure()
        #         plt.scatter(df_position[centroid_col[0]],
        #                     df_position[centroid_col[1]],
        #                     alpha=0.5,
        #                     color=colr,
        #                     label=f'{description} - P{position}, N={len(df_position)}')
        #         plt.title(f'Position: {position}')
        #         plt.xlabel(centroid_col[0])
        #         plt.ylabel(centroid_col[1])
        #         plt.legend()
        #         plt.show()
