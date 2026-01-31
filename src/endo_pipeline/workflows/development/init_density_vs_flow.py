import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from endo_pipeline.configs import get_datasets_in_collection, load_dataset_config
from endo_pipeline.library.analyze.integration.track_integration import (
    load_pc_diffae_liveseg_feats_merged_table,
)
from endo_pipeline.library.process.general_image_preprocessing import sequence_to_scalar
from endo_pipeline.settings.diffae_feature_dataframes import ColumnName

datasets = get_datasets_in_collection("live_cdh5_seg_based_feat_datasets")


dataset_info_cols = [
    ColumnName.DATASET.value,
    ColumnName.POSITION.value,
    ColumnName.TIMEPOINT.value,
]
density_cols = [
    "num_unique_tracks_before_filtering_at_T",
    "num_unique_tracks_after_filtering_at_T",
    "num_nuclei_in_crop",
    "total_nuclei_count_at_T",
]
cols_to_compute = dataset_info_cols + density_cols


summary_df = pd.DataFrame(columns=cols_to_compute)
for dataset_name in tqdm(datasets):
    config = load_dataset_config(dataset_name)
    if len(config.flow_conditions) != 1:
        print(f"Dataset {dataset_name} is not monoflow, skipping.")
        continue
    else:
        shear_stress = config.flow_conditions[0].shear_stress

    df_delayed = load_pc_diffae_liveseg_feats_merged_table(dataset_name)

    df = df_delayed[cols_to_compute].compute()
    # df = df[df.is_included]
    df = df.dropna(subset="total_nuclei_count_at_T")
    first_t = df[ColumnName.TIMEPOINT].min()
    df_first_t = df[df[ColumnName.TIMEPOINT] == first_t]

    groups = df_first_t.groupby(dataset_info_cols)

    for (ds, pos, tp), df_grp in groups:
        pos_num_nuclei = sequence_to_scalar(df_grp["total_nuclei_count_at_T"])
        pos_nuc_density = df_grp.num_nuclei_in_crop.mean()
        pos_num_seg_unfilt = sequence_to_scalar(df_grp["num_unique_tracks_before_filtering_at_T"])
        pos_num_seg_filt = sequence_to_scalar(df_grp["num_unique_tracks_after_filtering_at_T"])

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    {
                        ColumnName.DATASET.value: [ds],
                        ColumnName.POSITION.value: [pos],
                        ColumnName.TIMEPOINT.value: [tp],
                        "shear_stress": [shear_stress],
                        "total_nuclei_count_at_T": [pos_num_nuclei],
                        "num_unique_tracks_before_filtering_at_T": [pos_num_seg_unfilt],
                        "num_unique_tracks_after_filtering_at_T": [pos_num_seg_filt],
                        "num_nuclei_in_crop": [pos_nuc_density],
                    }
                ),
            ],
            ignore_index=True,
        )

for dens_col in density_cols:
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(
        data=summary_df,
        x="shear_stress",
        y=dens_col,
        hue=ColumnName.DATASET,
        marker="o",
        # s=20,
        ax=ax,
        legend=False,
    )
    ax.set_xlabel("Shear Stress (dyn/cm²)")
    # ax.set_ylabel("Cell Density (nuclei/FOV)")
