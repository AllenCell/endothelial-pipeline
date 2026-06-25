"""Methods for working with model manifests."""

import logging

from endo_pipeline.manifests import (
    ModelLocation,
    ModelManifest,
    get_model_manifest_dir,
    load_model_manifest,
)
from endo_pipeline.settings.literal_types import PatchTypeLiteral

logger = logging.getLogger(__name__)


def get_model_location_for_run(manifest: ModelManifest, run_name: str) -> ModelLocation:
    """Get the model location for the given run from the manifest, if it exists."""

    if run_name not in manifest.locations:
        logger.error(
            "Run [ %s ] does not have a location in model manifest [ %s ]",
            run_name,
            manifest.name,
        )
        raise KeyError(f"Unable to find run {run_name} in model manifest.")

    return manifest.locations[run_name]


def get_most_recent_run_name(model_manifest: ModelManifest) -> str:
    """Get the most recent run name from the model manifest."""
    available_runs = list(model_manifest.locations.keys())

    if not available_runs:
        logger.error("Model manifest [ %s ] has no runs", model_manifest.name)
        raise IndexError("No runs found in model manifest.")

    return available_runs[-1]


def get_feature_dataframe_manifest_name(
    model_manifest: ModelManifest,
    run_name: str | None,
    patch_type: PatchTypeLiteral = "grid_based",
) -> str:
    """Get the feature dataframe manifest name corresponding to model manifest and run name."""

    # Check that selected patch type is valid.
    if patch_type not in ("grid_based", "cell_centered"):
        logger.error("Patch type must be 'grid_based' or 'cell_centered'")
        raise ValueError(f"Patch type '{patch_type}' is not supported")

    # Default naming pattern is {model_manifest}_{run_name}_{patch_type}
    run_name_ = get_most_recent_run_name(model_manifest) if run_name is None else run_name
    feature_manifest_name = f"{model_manifest.name}_{run_name_}_{patch_type}"

    return feature_manifest_name


def get_model_manifest_with_parameters(
    workflow: str, parameters: dict | None = None
) -> ModelManifest:
    """Load model manifest with matching workflow containing given parameters."""

    manifests = []
    parameters = parameters or {}

    # Iterate through all model manifests to find ones with matching
    # workflow name and containing all given parameters.
    for manifest_file in get_model_manifest_dir().iterdir():
        manifest = load_model_manifest(manifest_file.stem)

        if manifest.workflow != workflow:
            continue

        if parameters.items() <= manifest.parameters.items():
            manifests.append(manifest)

    # If no manifests are found, raise error.
    if len(manifests) == 0:
        logger.error(
            "No model manifests found for workflow '%s' with parameters %s",
            workflow,
            parameters,
        )
        raise LookupError("Unable to find manifest with matching parameters")

    # If multiple manifests are found, then raise error. We could also instead
    # raise a warning and return the first manifest found.
    if len(manifests) > 1:
        logger.error(
            "Multiple model manifests found with given parameters: %s",
            " | ".join([manifest.name for manifest in manifests]),
        )
        raise ValueError("Found multiple manifests with matching parameters")

    return manifests[0]
