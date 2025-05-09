from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cellsmap.util.dataset_io import get_reference_datasets, load_config
from cellsmap.util.manifest_io import (
    get_diffae_manifest,
    get_feature_cols,
    get_track_diffae_manifest,
)
from cellsmap.util.manifest_preprocessing.manifest_pca import fit_pca
from cellsmap.util.set_output import get_output_path

out_dir = Path(get_output_path(Path(__file__).name, verbose=False))
data_config = load_config("data")
dataset_name_list = get_reference_datasets()

for dataset_name in dataset_name_list:
    # read in the grid crop-based diffae features
    diffae_grid_crops = get_diffae_manifest(dataset_name)

    # read in the segmentation-based diffae features
    diffae_tracking = get_track_diffae_manifest(dataset_name)

    # classic_tracking =
    classic_path = Path(
        r"C:\Users\serge.parent\Documents\projects\cellsmap\results\track_data_plots\segmentation_features_manifests\20241016_20X_segmentation_features.tsv"
    )
    classic_df = pd.read_csv(classic_path, sep="\t")

    integration_path = Path(
        r"C:\Users\serge.parent\Documents\projects\cellsmap\results\track_data_plots\single_cell_track_integration\20241016_20X_single_cell_track_integration.csv"
    )
    integration_df = pd.read_csv(integration_path)

    # fit the PCA (uses the reference datasets)
    pca = fit_pca(num_pcs=3)  # (only working with top 3 PCs)

    feat_cols = get_feature_cols(diffae_tracking)
    x_proj = pca.transform(diffae_tracking[feat_cols].values)

    # add PCA components to dataframe
    for pc in range(3):
        diffae_tracking[f"pc{pc+1}"] = x_proj[:, pc]

    for i, feat in enumerate(diffae_df.columns):
        if f"feat_{i}" in feat:
            fig, ax = plt.subplots()
            sns.lineplot(data=diffae_df, x="frame_number", y=feat, label="DiffAE")
            fig.suptitle(f"DiffAE {feat}")
            fig.savefig(out_dir / f"diffae_{feat}.png")
            plt.close(fig)
            # break

            # diffae_df[feat] = diffae_df[feat].astype(float)
            # classic_df[feat] = classic_df[feat].astype(float)
            # integration_df[feat] = integration_df[feat].astype(float)
