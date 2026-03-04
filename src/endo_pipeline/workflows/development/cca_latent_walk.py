"""CCA latent walk: walk along the CCA axis in PC space and generate images.

This workflow:
1. Loads all datasets, fits PCA, merges optical-flow features, and projects
   onto the saved CCA versor to get R = X_pc @ ŵ_cca for every crop.
2. Computes walk bounds as mean(R) ± sigma·std(R).
3. Decomposes R into per-PC variance contributions to identify the driving PCs.
4. Generates latent-walk images along the CCA axis using:
   (a) all 80 PCs  (full CCA versor)
   (b) only the top-3 PCs (sparse CCA direction, zeroing out all but pc_1 to pc_3)
"""

import logging

from endo_pipeline.cli import tags
from endo_pipeline.settings.workflow_defaults import (
    CCA_WEIGHTS_LOCATION_KEY,
    CCA_WEIGHTS_MANIFEST_NAME,
    DEFAULT_MODEL_MANIFEST_NAME,
    DEFAULT_MODEL_RUN_NAME,
    DEFAULT_PCA_DATASET_COLLECTION_NAME,
    OPTICAL_FLOW_FEATURE,
    OPTICAL_FLOW_MANIFEST_NAME,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

TAGS = ["diffae", "cca_interpretation", tags.GPU]


def main(
    model_manifest_name: str = DEFAULT_MODEL_MANIFEST_NAME,
    run_name: str = DEFAULT_MODEL_RUN_NAME,
    dataset_collection: str = DEFAULT_PCA_DATASET_COLLECTION_NAME,
    num_pcs: int = 80,
    n_steps: int = 7,
    sigma: float = 3.0,
    num_seeds: int = 5,
    random_seed: int = RANDOM_SEED,
) -> None:
    r"""
    Walk along the CCA axis in PC space and generate synthetic images.

    The CCA axis is the direction in PC space that is maximally correlated with
    the optical-flow based coherence feature.  Walking along this axis produces images
    that transition from low to high migration coherence (hopefully!)

    Two walks are produced:

    - **all PCs**: uses the full 80-dimensional CCA versor.
    - **top 3 PCs**: zeroes out all but pc_1 to pc_3 (no re-normalisation),
      so R values are on the same scale as the all-PC walk and the two
      are directly comparable.

    Parameters
    ----------
    model_manifest_name
        Model manifest for loading the DiffAE model.
    run_name
        MLflow run name within the model manifest.
    dataset_collection
        Dataset collection to load and project.
    num_pcs
        Number of principal components to use for PCA fitting.
    n_steps
        Number of discrete steps in each latent walk.
    sigma
        Walk range in standard deviations: mean(R) ± sigma·std(R).
    num_seeds
        Number of noise seeds for image generation; results are averaged.
    random_seed
        Base random seed for reproducibility.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import get_datasets_in_collection
    from endo_pipeline.io import get_output_path, load_dataframe, load_model
    from endo_pipeline.library.analyze.diffae_dataframe_utils import (
        fit_pca,
        get_dataframe_for_dynamics_workflows,
    )
    from endo_pipeline.library.model.diffae import generate_from_coords
    from endo_pipeline.manifests import (
        get_dataframe_location_for_dataset,
        get_feature_dataframe_manifest_name,
        load_dataframe_manifest,
        load_model_manifest,
    )
    from endo_pipeline.settings import DIFFAE_PC_COLUMN_NAMES

    np.random.seed(random_seed)
    output_dir = get_output_path("cca_latent_walk")
    pc_columns = DIFFAE_PC_COLUMN_NAMES[:num_pcs]

    # ──────────────────────────────────────────────────────────────────────
    # Load manifests, fit PCA, gather data
    # ──────────────────────────────────────────────────────────────────────
    model_manifest = load_model_manifest(model_manifest_name)
    dataframe_manifest_name = get_feature_dataframe_manifest_name(
        model_manifest, run_name, crop_pattern="grid"
    )
    dataframe_manifest = load_dataframe_manifest(dataframe_manifest_name)
    pca = fit_pca(num_pcs=num_pcs)

    datasets = get_datasets_in_collection(dataset_collection)
    dataframe_manifest_optical_flow = load_dataframe_manifest(OPTICAL_FLOW_MANIFEST_NAME)

    df_all_list = []
    for dataset_name in datasets:
        logger.info("Loading %s", dataset_name)
        df_ds = get_dataframe_for_dynamics_workflows(
            dataset_name, dataframe_manifest, pca=pca, filter_dataframe=True
        )
        of_loc = get_dataframe_location_for_dataset(dataframe_manifest_optical_flow, dataset_name)
        df_of = load_dataframe(of_loc)
        df_merged = df_ds.merge(
            df_of,
            on=["dataset", "position", "frame_number", "start_x", "start_y"],
            how="inner",
            suffixes=("", "_optical_flow"),
        )
        df_all_list.append(df_merged)

    df_all = pd.concat(df_all_list, ignore_index=True)
    del df_all_list

    # ──────────────────────────────────────────────────────────────────────
    # Load CCA weights and compute R = X_pc @ ŵ_cca
    # ──────────────────────────────────────────────────────────────────────
    cca_manifest = load_dataframe_manifest(CCA_WEIGHTS_MANIFEST_NAME)
    cca_location = cca_manifest.locations[CCA_WEIGHTS_LOCATION_KEY]
    cca_weights_df = load_dataframe(cca_location)

    cca_features = cca_weights_df["input_feature"].tolist()
    cca_weights = cca_weights_df["weight"].to_numpy()
    cca_versor = cca_weights / np.linalg.norm(cca_weights)

    # Project onto the versor so R and the walk direction are self-consistent:
    #   R = X @ ŵ  and  p = R · ŵ  ⟹  p @ ŵ = R  ✓
    df_all_clean = df_all.dropna(subset=pc_columns + [OPTICAL_FLOW_FEATURE])
    X_pc = df_all_clean[cca_features].to_numpy()
    R = X_pc @ cca_versor

    mu_R = float(R.mean())
    std_R = float(R.std())
    R_lo = mu_R - sigma * std_R
    R_hi = mu_R + sigma * std_R

    logger.info("CCA projection  R:  mean=%.4f  std=%.4f", mu_R, std_R)
    logger.info("Walk bounds [%.1f sigma]:  [%.4f,  %.4f]", sigma, R_lo, R_hi)

    # ──────────────────────────────────────────────────────────────────────
    # Per-PC contribution analysis
    # ──────────────────────────────────────────────────────────────────────
    # Var(R) = Σ_j Var(x_j) · ŵ_j²   (PCs are orthogonal → additive)
    mean_contrib = X_pc.mean(axis=0) * cca_versor
    var_contrib = X_pc.var(axis=0) * cca_versor**2
    total_var_R = var_contrib.sum()
    frac_var = var_contrib / total_var_R

    df_pc_importance = pd.DataFrame(
        {
            "pc": cca_features,
            "cca_weight": cca_weights,
            "cca_versor_component": cca_versor,
            "mean_contribution_to_R": mean_contrib,
            "variance_contribution_to_R": var_contrib,
            "fraction_of_var_R": frac_var,
        }
    ).sort_values("fraction_of_var_R", ascending=False)

    csv_path = Path(output_dir) / "cca_per_pc_importance.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_pc_importance.to_csv(csv_path, index=False)
    logger.info(
        "Top PCs contributing to Var(R):\n%s",
        df_pc_importance.head(10).to_string(index=False),
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(df_pc_importance["pc"], df_pc_importance["fraction_of_var_R"])
    ax.set_ylabel("Fraction of Var(R)")
    ax.set_title("Per-PC contribution to CCA projection variance")
    ax.set_xticks(range(len(df_pc_importance)))
    ax.set_xticklabels(df_pc_importance["pc"], rotation=45, ha="right", fontsize=6)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "cca_per_pc_variance_fraction.png", dpi=150)
    plt.show()
    plt.close(fig)

    # ──────────────────────────────────────────────────────────────────────
    # Build walk coordinates in PC space
    # ──────────────────────────────────────────────────────────────────────
    walk_R_values = np.linspace(R_lo, R_hi, n_steps)

    # (a) All PCs
    walk_pca_all = np.array([r * cca_versor for r in walk_R_values])
    walk_diffae_all = pca.inverse_transform(walk_pca_all)

    # (b) Top 3 PCs only — zero out everything else, keep original scale
    # (no re-normalisation so R_top3 is the partial sum of the same dot
    #  product as R_all and the two walks are directly comparable)
    top3_mask = np.array([f in ("pc_1", "pc_2", "pc_3") for f in cca_features])
    cca_versor_top3 = np.where(top3_mask, cca_versor, 0.0)

    R_top3 = X_pc @ cca_versor_top3
    mu_R3 = float(R_top3.mean())
    std_R3 = float(R_top3.std())
    R3_lo = mu_R3 - sigma * std_R3
    R3_hi = mu_R3 + sigma * std_R3
    walk_R3_values = np.linspace(R3_lo, R3_hi, n_steps)

    walk_pca_top3 = np.array([r * cca_versor_top3 for r in walk_R3_values])
    walk_diffae_top3 = pca.inverse_transform(walk_pca_top3)

    logger.info("Top-3 CCA projection:  mean=%.4f  std=%.4f", mu_R3, std_R3)
    logger.info("Top-3 walk bounds [%.1f sigma]:  [%.4f,  %.4f]", sigma, R3_lo, R3_hi)

    # ──────────────────────────────────────────────────────────────────────
    # Generate latent-walk images
    # ──────────────────────────────────────────────────────────────────────
    model = load_model(model_manifest.locations[run_name], instantiate=True)

    def _run_walk_and_save(
        walk_diffae: np.ndarray,
        walk_R_vals: np.ndarray,
        label: str,
    ) -> None:
        """Generate images for a latent walk, save per-seed and mean strips."""
        walk_dir = Path(output_dir) / label
        walk_dir.mkdir(parents=True, exist_ok=True)

        walk_imgs = []
        seeds = [random_seed] + np.random.randint(0, 999, num_seeds - 1).tolist()
        for seed_id, seed in enumerate(seeds):
            walk_img = generate_from_coords(
                model, walk_diffae, n_noise_samples=1, num_gpus=NUM_GPUS, random_seed=seed
            )
            walk_imgs.append(walk_img)

            fig, axs = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
            for i in range(n_steps):
                axs[i].imshow(walk_img[i], cmap="gray", vmin=-1, vmax=0.5)
                axs[i].set_title(f"R={walk_R_vals[i]:.2f}", fontsize=7)
                axs[i].axis("off")
            fig.suptitle(f"{label}  seed={seed}", fontsize=9)
            plt.tight_layout()
            fig.savefig(walk_dir / f"walk_{seed_id}.png", dpi=150)
            plt.close(fig)

        walk_imgs_mean = np.array(walk_imgs).mean(axis=0)
        fig, axs = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
        for i in range(n_steps):
            axs[i].imshow(walk_imgs_mean[i], cmap="viridis", vmin=-1, vmax=0.5)
            axs[i].set_title(f"R={walk_R_vals[i]:.2f}", fontsize=7)
            axs[i].axis("off")
        fig.suptitle(f"{label}  mean over {num_seeds} seeds", fontsize=9)
        plt.tight_layout()
        fig.savefig(walk_dir / "walk_mean_all_seeds.png", dpi=150)
        plt.close(fig)

    logger.info("=== CCA latent walk - all %d PCs ===", num_pcs)
    _run_walk_and_save(walk_diffae_all, walk_R_values, label="cca_walk_all_pcs")

    logger.info("=== CCA latent walk - top 3 PCs only ===")
    _run_walk_and_save(walk_diffae_top3, walk_R3_values, label="cca_walk_top3_pcs")

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
