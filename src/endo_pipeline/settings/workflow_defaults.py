"""Workflow default settings."""

DEFAULT_MODEL_MANIFEST_NAME: str = "diffae_04_10"
"""Default model manifest for loading models and model features."""

DEFAULT_MODEL_RUN_NAME: str | None = None
"""Default model run name within the default model manifest."""

DEFAULT_COLLECTION_FOR_DIFFAE_EVALUATION: str = "live_20X_objective_3i_microscope"
"""Default dataset collection on which to evaluate a trained Diff AE model."""
