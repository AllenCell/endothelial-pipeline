"""Methods for working with model manifests."""

import logging

from endo_pipeline.manifests import ModelLocation, ModelManifest

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
