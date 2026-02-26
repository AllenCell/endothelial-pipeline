import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import numpy as np
np.random.seed(666)
from pathlib import Path
import matplotlib.pyplot as plt

from endo_pipeline.io import get_output_path, load_model
from endo_pipeline.library.analyze.diffae_dataframe_utils import fit_pca
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    get_most_recent_run_name,
    load_dataframe_manifest,
    load_model_manifest,
)

from endo_pipeline.library.model.diffae import generate_from_coords
from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
)

output_dir = get_output_path("mig_lda_latent_walk")

model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
dataframe_manifest_name = get_feature_dataframe_manifest_name(
    model_manifest, DEFAULT_MODEL_RUN_NAME, crop_pattern="grid"
)
dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
pca, pca_input_dataframe = fit_pca(num_pcs=80, return_pca_input_dataframe=True)

lda_info_path = "/allen/aics/users/matheus.viana/repos/endothelial-pipeline/results/2026-02-19/find_coherent_mig/lda_transform_pcs_only.json"
with open(lda_info_path, "r") as f:
    lda_info = json.load(f)

pca_transformed_data = pca.transform(pca_input_dataframe.values)
lda_weights = np.array(lda_info["weights"])
lda_versor = lda_weights / np.linalg.norm(lda_weights)
lda_intercept = np.array(lda_info["intercept"])

lda_projected_data = pca_transformed_data @ lda_versor

nsteps = 7
sigma = 3.0
mu = np.mean(lda_projected_data)
std = np.std(lda_projected_data)
start_val = mu - sigma * std
end_val   = mu + sigma * std
walk_latent = np.linspace(start_val, end_val, nsteps)
walk_pca_space = [p * lda_versor for p in walk_latent]
walk_pca_space = np.array(walk_pca_space)
walk_diffae_space = pca.inverse_transform(walk_pca_space)


model_manifest = load_model_manifest(DEFAULT_MODEL_MANIFEST_NAME)
model = load_model(model_manifest.locations[DEFAULT_MODEL_RUN_NAME], instantiate=True)

NSEEDS = 50

def main():

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

if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)

# from pacmap import PaCMAP
# reducer = PaCMAP(n_components=2, n_neighbors=30, MN_ratio=1.5, FP_ratio=2.0, save_pairs=True, save_tree=True)
# emb_data = reducer.fit_transform(pca_transformed_data[::10])
# emb_walk = reducer.transform(walk_pca_space)
# plt.figure(figsize=(10, 10))
# plt.scatter(emb_data[:, 0], emb_data[:, 1], alpha=0.1, s=5)
# plt.plot(emb_walk[:, 0], emb_walk[:, 1], color="red", lw=5)
# plt.scatter(emb_walk[:, 0], emb_walk[:, 1], color="red", s=100)
# plt.tight_layout()
# plt.savefig(Path(output_dir) / "lda_walk_pacmap_projection.png")
# plt.close()
