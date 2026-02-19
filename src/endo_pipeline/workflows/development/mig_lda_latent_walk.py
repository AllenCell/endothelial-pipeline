import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from endo_pipeline.io import get_output_path, load_model
from endo_pipeline.library.analyze.diffae_dataframe_utils import (
    fit_pca,
    get_dataframe_for_dynamics_workflows,
)
from endo_pipeline.manifests import (
    get_feature_dataframe_manifest_name,
    get_most_recent_run_name,
    load_dataframe_manifest,
    load_model_manifest,
)

from endo_pipeline.library.model.diffae import DiffusionAutoEncoder, generate_from_coords

from endo_pipeline.settings.diffae_feature_dataframes import DIFFAE_PC_COLUMN_NAMES
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

run_name = DEFAULT_MODEL_RUN_NAME
model_manifest_name = DEFAULT_MODEL_MANIFEST_NAME
model_manifest = load_model_manifest(model_manifest_name)
run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
model = load_model(model_manifest.locations[run_name_], instantiate=True)
walk_img = generate_from_coords(model, walk_diffae_space, n_noise_samples=1, num_gpus=1, random_seed=42)

import pdb; pdb.set_trace()