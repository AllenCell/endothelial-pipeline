import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

from endo_pipeline.configs import get_datasets_in_collection
from endo_pipeline.io import (
    load_dataframe,
    get_output_path,
    load_model,
)
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    get_dataframe_location_for_dataset,
    get_feature_dataframe_manifest_name,
    load_dataframe_manifest,
    load_model_manifest,
)
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

from endo_pipeline.library.model.diffae import generate_from_coords


output_dir = get_output_path("pc_mig_cca")

lda_dataframe_manifest = load_dataframe_manifest("lda_weights")
lda_location = get_dataframe_location_for_dataset(lda_dataframe_manifest, "80_pcs")
df_lda = load_dataframe(lda_location)

model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca = fit_pca(num_pcs=80)

dataframe_manifest_optical_flow = load_dataframe_manifest("optical_flow")

datasets = get_datasets_in_collection("diffae_model_training")

def walk_along_versor(versor, X, nsteps=7, sigma=3.0):
    mu = X.mean(axis=0, keepdims=True)
    projected_X = (X - mu) @ versor
    std = np.std(projected_X)
    steps = np.linspace(-sigma*std, sigma*std, nsteps)
    walk = mu + steps[:, None] * versor[None, :]
    return walk

def walk_latent_space_along_canonical_axis(cca, X, nsteps=7, sigma=3.0):
    cca_weights = cca.x_weights_[:, 0]
    cca_versor = cca_weights / np.linalg.norm(cca_weights)

    walk_pca_space = walk_along_versor(cca_versor, X, nsteps=nsteps, sigma=sigma)

    # cca_projected_data = X @ cca_versor

    # nsteps = 7
    # sigma = 3.0
    # mu = np.mean(cca_projected_data)
    # std = np.std(cca_projected_data)
    # start_val = mu - sigma * std
    # end_val   = mu + sigma * std
    # walk_latent = np.linspace(start_val, end_val, nsteps)
    # walk_pca_space = [p * cca_versor for p in walk_latent]
    # walk_pca_space = np.array(walk_pca_space)
    walk_diffae_space = pca.inverse_transform(walk_pca_space)

    model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
    model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)

    NSEEDS = 50

    walk_imgs = []
    for seed_id, seed in enumerate([666] + np.random.randint(0, 999, NSEEDS-1).tolist()):
        walk_img = generate_from_coords(model, walk_diffae_space, n_noise_samples=1, num_gpus=1, random_seed=seed)
        walk_imgs.append(walk_img)

        fig, axs = plt.subplots(1, nsteps, figsize=(nsteps*2, 2))
        for i in range(nsteps): axs[i].imshow(walk_img[i], cmap="gray", vmin=-1, vmax=0.5); axs[i].axis("off")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"lda_walk_{seed_id}.png")
        plt.close()

    walk_imgs_mean = np.array(walk_imgs).mean(axis=0)
    fig, axs = plt.subplots(1, nsteps, figsize=(nsteps*2, 2))
    for i in range(nsteps): axs[i].imshow(walk_imgs_mean[i], cmap="viridis", vmin=-1, vmax=0.5); axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"lda_walk_all_seeds.png")
    plt.close()


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    df_proj_full_list = []

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        df_dataset = get_dataframe_for_dynamics_workflows(
            dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
        )

        optical_flow_location = get_dataframe_location_for_dataset(
            dataframe_manifest_optical_flow, dataset_name
        )
        df_optical_flow = load_dataframe(optical_flow_location)

        df_dataset = df_dataset.merge(
            df_optical_flow,
            on=["dataset", "position", "frame_number", "start_x", "start_y"],
            how="inner",
            suffixes=("", "_optical_flow"),
        )

        df_proj_full_list.append(df_dataset)

    df = pd.concat(df_proj_full_list, ignore_index=True)
    df = df.dropna(subset=["optical_flow_angle_std"])

    input_features = [col for col in df.columns if col.startswith("pc_")]
    for target_feature in [col for col in df.columns if "optical_flow" in col]:
        X = df[input_features].values
        y = df[target_feature].values
        cca = CCA(n_components=1, max_iter=5000, tol=1e-12)
        Xc, yc = cca.fit_transform(X,y)
        corr = float(np.corrcoef(Xc[:,0], yc[:,0])[0,1])
        print(f"Canonical Correlation with {target_feature} = {corr:.3f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2]})
        xdata, ydata = Xc[:,0], yc[:,0]
        xmin, xmax = np.percentile(xdata, [1, 99])
        ymin, ymax = np.percentile(ydata, [1, 99])
        density = ax1.hexbin(
            xdata,
            ydata,
            gridsize=80,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        cbar = fig.colorbar(density, ax=ax1, pad=0.01)
        cbar.set_label("Point density (log scale)")
        ax1.set_xlabel("Canonical Variable 1 (PCs)")
        ax1.set_ylabel(f"Canonical Variable 1 ({target_feature})")
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)

        ax2.bar(input_features, cca.x_weights_[:,0])
        ax2.set_xticks(range(len(input_features)))
        ax2.set_xticklabels(input_features, rotation=45, ha="right", fontsize=6)
        fig.suptitle(f"CCA Projection vs {target_feature}, Canonical Correlation = {corr:.3f}")
        fig.tight_layout()
        plt.savefig(output_dir / f"cca_weights_{target_feature}.png")
        plt.close()

    target_feature = "optical_flow_angle_std"
    X = df[input_features].values
    y = df[target_feature].values
    cca = CCA(n_components=1, max_iter=5000, tol=1e-12)
    cca.fit(X, y)

    walk_latent_space_along_canonical_axis(cca, X, nsteps=7, sigma=3.0)

if __name__ == "__main__":
    from    .cli import workflow_cli

    workflow_cli(main)