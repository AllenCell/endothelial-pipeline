"""Model QC workflow for Diffusion Autoencoder evaluation."""

import logging
from typing import Literal

from endo_pipeline.settings.workflow_defaults import (
    DEFAULT_MODEL_QC_MANIFEST_NAMES,
    DEFAULT_MODEL_QC_RUN_NAMES,
    MODEL_QC_NOISE_LEVELS,
    RANDOM_SEED,
)


def main(
    model_manifest_name: list[str] = DEFAULT_MODEL_QC_MANIFEST_NAMES,
    run_name: list[str] | None = DEFAULT_MODEL_QC_RUN_NAMES,
    mode: Literal["basic", "comparison"] | None = None,
    random_seed: int = RANDOM_SEED,
    include_negative_controls: bool | None = None,
    save_intermediate_plots: bool | None = None,
    save_crops_as_tiff: bool = False,
    compute_metrics: bool | None = None,
    compute_baseline: bool = True,
    num_seeds: int = 1,
) -> None:
    r"""
    Run quality check assessment for trained Diffusion Autoencoder models.

    #diffae #test-ready #gpu

    This workflow combines:

    - Basic model QC with denoising visualization and negative controls
    - Multi-model comparison with quantitative metrics (correlation, SSIM, LPIPS)
    - Support for single or multiple random seeds for robustness analysis

    It supports two main modes:

    - **basic** : single model evaluation with visual QC and optional
      negative controls.
    - **comparison** : multi-model evaluation with quantitative metrics
      and comparison plots.

    Usage Modes
    -----------
    1. Basic QC (single model, visual checks with negative controls)::

        endopipe model-qc --model_manifest_name model1

    2. Comparison mode (multiple models, metrics)::

        endopipe model-qc --mode comparison \\
            --model_manifest_name model1 --model_manifest_name model2

    3. Full analysis (comparison + negative controls + intermediate plots)::

        endopipe model-qc --mode comparison \\
            --model_manifest_name model1 --model_manifest_name model2 \\
            --include_negative_controls --save_intermediate_plots

    Mode-Dependent Defaults
    -----------------------
    All flags below can be explicitly overridden in either mode.  When set
    to ``None`` (the default), they resolve based on the active mode:

    | Flag                          | Basic (1 model)                      | Comparison (2+ models)                |
    | ----------------------------- | ------------------------------------ | ------------------------------------- |
    | ``--compute-metrics``           | ``False`` (opt-in)                     | ``True`` (always)                       |
    | ``--include-negative-controls`` | ``True`` (unless ``--compute-metrics``)  | ``False`` (opt-in)                      |
    | ``--save-intermediate-plots``   | ``True`` (always)                      | ``False`` (opt-in)                      |
    | ``--save-crops-as-tiff``        | ``False`` (opt-in)                     | ``False`` (opt-in)                      |
    | ``--compute-baseline``          | ``False`` unless ``--compute-metrics`` | ``True`` unless ``--no-compute_baseline`` |

    Parameters
    ----------
    model_manifest_name
        Model manifest name(s).  Provide once per model.
    run_name
        MLflow run name(s), one per manifest.  ``None`` → most recent.
    mode
        ``"basic"`` or ``"comparison"``.  Auto-detected from the number of
        models when omitted.
    random_seed
        Seed for noise generation.  Centre of the seed range when
        *num_seeds* > 1.
    include_negative_controls
        Generate scrambled-embedding / scrambled-input negative controls.
    save_intermediate_plots
        Save per-example contact sheets.
    save_crops_as_tiff
        Save individual crops as TIFF files (default seed only).
    compute_metrics
        Compute Pearson correlation, SSIM, and LPIPS metrics.
    compute_baseline
        Compute next-timepoint baseline metrics.
    num_seeds
        Number of seeds to evaluate.  Results are averaged when > 1.

    Examples
    --------
    Default: 10-model latent dimension comparison study:
        endopipe model-qc --mode comparison

    Basic QC for a single model:
        endopipe model-qc --model_manifest_name my_model --run_name my_run

    Compare multiple models with metrics:
        endopipe model-qc --model_manifest_name model1 --model_manifest_name model2 \\
            --run_name run1 --run_name run2

    Full analysis with all features (default 10 models):
        endopipe model-qc --mode comparison --include_negative_controls --save_intermediate_plots --num_seeds 10
    """
    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.library.model.model_qc import (
        ModelKey,
        aggregate_seed_metrics,
        build_models_data,
        compute_baseline_data,
        create_comparison_plots_and_summary,
        evaluate_single_model,
    )
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest
    from endo_pipeline.settings.examples import (
        MODEL_QC_EXAMPLES_REP_2_POSITIONS,
        MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
        MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
    )

    logger = logging.getLogger(__name__)

    model_manifest_names = list(model_manifest_name)

    if run_name is None:
        run_names = [None] * len(model_manifest_names)
    else:
        run_names = list(run_name)

    # Ensure we have matching numbers of manifests and run names
    if len(run_names) != len(model_manifest_names):
        raise ValueError(
            f"Number of run_names ({len(run_names)}) must match "
            f"number of model_manifest_names ({len(model_manifest_names)})"
        )

    # Determine mode
    if mode is None:
        mode = "comparison" if len(model_manifest_names) > 1 else "basic"
    is_comparison_mode = mode == "comparison"

    # Apply mode-dependent defaults
    if compute_metrics is None:
        compute_metrics = is_comparison_mode
    if include_negative_controls is None:
        # In basic mode, skip negative controls when computing metrics (expensive
        # GPU work with no metric benefit). Still on by default for visual-only QC.
        include_negative_controls = not is_comparison_mode and not compute_metrics
    if save_intermediate_plots is None:
        save_intermediate_plots = not is_comparison_mode

    # Baseline only relevant when computing metrics
    if not compute_metrics:
        compute_baseline = False

    logger.info(f"Running model_qc in '{mode}' mode")
    logger.info(f"Models: {len(model_manifest_names)}, Compute metrics: {compute_metrics}")
    logger.info(f"Negative controls: {include_negative_controls}")
    logger.info(f"Save intermediate plots: {save_intermediate_plots}")

    # Set up example sets
    # For comparison mode, exclude training positions (only validation + rep2)
    # For basic mode, include all positions
    if is_comparison_mode:
        example_sets_all = [
            (MODEL_QC_EXAMPLES_VALIDATION_POSITIONS, "validation_positions"),
            (MODEL_QC_EXAMPLES_REP_2_POSITIONS, "rep_2_positions"),
        ]
    else:
        example_sets_all = [
            (MODEL_QC_EXAMPLES_TRAINING_POSITIONS, "training_positions"),
            (MODEL_QC_EXAMPLES_VALIDATION_POSITIONS, "validation_positions"),
            (MODEL_QC_EXAMPLES_REP_2_POSITIONS, "rep_2_positions"),
        ]

    # Define the example sets for the metrics
    example_sets_for_metrics = {"validation_positions", "rep_2_positions"}

    if DEMO_MODE:
        logger.info("DEMO MODE: Limiting to first example of each set")
        example_sets_all = [(examples[:1], label) for examples, label in example_sets_all]

    # Generate list of seeds to evaluate
    if num_seeds == 1:
        seeds_to_evaluate = [random_seed]
    else:
        half_range = num_seeds // 2
        seeds_to_evaluate = list(
            range(random_seed - half_range, random_seed - half_range + num_seeds)
        )

    logger.info(f"Evaluating with {len(seeds_to_evaluate)} seed(s): {seeds_to_evaluate}")
    logger.info(f"Default seed for saving crops/plots: {random_seed}")

    # Eagerly resolve any None run names to the most-recent run, so each
    # model has a stable ``ModelKey``.  This is the single source of truth
    # used for dict keys, output paths, and plot labels.
    model_keys: list[ModelKey] = []
    for manifest_name, run_name_input in zip(model_manifest_names, run_names, strict=True):
        if run_name_input is None:
            resolved_run_name = get_most_recent_run_name(load_model_manifest(manifest_name))
        else:
            resolved_run_name = run_name_input
        model_keys.append(ModelKey(manifest_name, resolved_run_name))

    # Reject duplicate (manifest, run) pairs: they would collide on output
    # paths and collapse silently in downstream dict-keyed structures.
    duplicates = [key for key in model_keys if model_keys.count(key) > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate (manifest_name, run_name) entries are not allowed: "
            f"{sorted(set(duplicates))}"
        )

    # Storage for all results across seeds
    # Structure: {ModelKey: {seed: {example_set_label: [per-example metrics]}}}
    all_seed_results: dict[ModelKey, dict[int, dict[str, list[dict]]]] = {}

    logger.info("Running model evaluations...")
    for model_key in model_keys:
        all_seed_results[model_key] = {}
        for seed in seeds_to_evaluate:
            is_default = seed == random_seed
            result = evaluate_single_model(
                model_key=model_key,
                random_seed=seed,
                example_sets_all=example_sets_all,
                example_sets_for_metrics=example_sets_for_metrics,
                save_intermediate_plots=save_intermediate_plots,
                save_crops_as_tiff=save_crops_as_tiff,
                include_negative_controls=include_negative_controls,
                compute_metrics=compute_metrics,
                noise_levels=MODEL_QC_NOISE_LEVELS,
                compute_baseline=compute_baseline,
                is_default_seed=is_default,
                num_gpus=NUM_GPUS,
            )
            all_seed_results[model_key][seed] = result

    # Aggregate metrics and create plots / summary
    if compute_metrics:
        logger.info("Aggregating metrics across seeds...")

        all_metrics, _ = aggregate_seed_metrics(
            all_seed_results, model_keys, example_sets_for_metrics, seeds_to_evaluate
        )
        baseline_data = compute_baseline_data(all_metrics, compute_baseline)
        models_data = build_models_data(all_metrics, model_keys, baseline_data, compute_baseline)

        if is_comparison_mode:
            logger.info("Creating comparison plots across models...")
        create_comparison_plots_and_summary(
            models_data,
            model_keys,
            seeds_to_evaluate,
            baseline_data,
            compute_baseline,
        )

    logger.info("Model QC workflow completed successfully!")


if __name__ == "__main__":

    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
