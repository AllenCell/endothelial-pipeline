"""Single-model qualitative DiffAE QC: denoising contact sheet + controls."""

import logging

from endo_pipeline.settings.workflow_defaults import RANDOM_SEED


def main(
    model_manifest_name: str,
    run_name: str | None = None,
    random_seed: int = RANDOM_SEED,
    include_negative_controls: bool = True,
    save_intermediate_plots: bool = True,
    save_crops_as_tiff: bool = False,
) -> None:
    r"""
    Run single-model qualitative QC for a trained Diffusion Autoencoder.

    #diffae #gpu

    Runs denoising for ONE model and renders the visual QC contact sheet
    (conditioning encoder input / target / denoised, plus the scrambled-latent
    and scrambled-input negative controls). This is the qualitative
    "does this model work, and do the controls break it as expected?" check.

    Cross-model quantitative comparisons live in the separate
    ``visualize-diffae-model-comparison`` workflow.

    Parameters
    ----------
    model_manifest_name
        Model manifest name to evaluate.
    run_name
        MLflow run name. ``None`` resolves to the most recent run.
    random_seed
        Seed for noise generation.
    include_negative_controls
        Generate the scrambled-embedding / scrambled-input negative controls.
    save_intermediate_plots
        Save the per-example denoising contact sheets.
    save_crops_as_tiff
        Also save individual crops as TIFF files.
    """
    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.library.model.model_qc import ModelKey, evaluate_single_model
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest
    from endo_pipeline.settings.examples import (
        MODEL_QC_EXAMPLES_REP_2_POSITIONS,
        MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
        MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
    )
    from endo_pipeline.settings.workflow_defaults import MODEL_QC_NOISE_LEVELS

    logger = logging.getLogger(__name__)

    if run_name is None:
        run_name = get_most_recent_run_name(load_model_manifest(model_manifest_name))
    model_key = ModelKey(model_manifest_name, run_name)
    logger.info("Running single-model QC for %s / %s", model_manifest_name, run_name)

    # Basic-mode example roster: all positions (training + validation + rep-2).
    example_sets_all = [
        (MODEL_QC_EXAMPLES_TRAINING_POSITIONS, "training_positions"),
        (MODEL_QC_EXAMPLES_VALIDATION_POSITIONS, "validation_positions"),
        (MODEL_QC_EXAMPLES_REP_2_POSITIONS, "rep_2_positions"),
    ]
    if DEMO_MODE:
        logger.info("DEMO MODE: limiting to the first example of each set")
        example_sets_all = [(examples[:1], label) for examples, label in example_sets_all]

    # Qualitative QC only: no metrics, no baseline, single (default) seed.
    evaluate_single_model(
        model_key=model_key,
        random_seed=random_seed,
        example_sets_all=example_sets_all,
        example_sets_for_metrics={"validation_positions", "rep_2_positions"},
        save_intermediate_plots=save_intermediate_plots,
        save_crops_as_tiff=save_crops_as_tiff,
        include_negative_controls=include_negative_controls,
        compute_metrics=False,
        noise_levels=MODEL_QC_NOISE_LEVELS,
        compute_baseline=False,
        is_default_seed=True,
        num_gpus=NUM_GPUS,
    )
    logger.info("Single-model QC complete for %s / %s", model_manifest_name, run_name)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
