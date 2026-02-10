"""
Unified Model QC workflow for Diffusion Autoencoder model evaluation.

This module combines the functionality of:
- Basic model QC with denoising visualization and negative controls
- Multi-model comparison with quantitative metrics (correlation, SSIM, LPIPS)
- Support for single or multiple random seeds for robustness analysis

Usage modes:
-----------
1. Basic QC (single model, visual checks with negative controls):
   endopipe model_qc --model_manifest_name model1

2. Comparison mode (multiple models, metrics):
   endopipe model_qc --mode comparison \\
       --model_manifest_name model1 --model_manifest_name model2

3. Full analysis (comparison + negative controls + intermediate plots):
   endopipe model_qc --mode comparison \\
       --model_manifest_name model1 --model_manifest_name model2 \\
       --include_negative_controls --save_intermediate_plots
"""

import numpy as np

from endo_pipeline.cli import tags
from endo_pipeline.settings.workflow_defaults import RANDOM_SEED

TAGS = ["diffae", tags.TEST_READY, tags.GPU]

# =============================================================
# Default model configurations for latent dimension comparison 
# =============================================================

# 10 models: 8 BF latent dims (8-1024) + 2 CDH5 conditioned (512, 1024)
DEFAULT_MODEL_MANIFEST_NAMES = [
    "diffae_baseline_exclude_cell_piling",  # 8 BF
    "diffae_baseline_exclude_cell_piling",  # 16 BF
    "diffae_baseline_exclude_cell_piling",  # 32 BF
    "diffae_baseline_exclude_cell_piling",  # 64 BF
    "diffae_baseline_exclude_cell_piling",  # 128 BF
    "diffae_baseline_exclude_cell_piling",  # 256 BF
    "diffae_baseline_exclude_cell_piling",  # 512 BF
    "diffae_baseline_exclude_cell_piling",  # 1024 BF
    "diffae_cdh5_conditioned",              # 512 CDH5
    "diffae_cdh5_conditioned",              # 1024 CDH5
]

DEFAULT_RUN_NAMES = [
    "20260207_latent_8",
    "20260205_latent_16",
    "20260203_latent_32",
    "20260206_latent_64",
    "20260127_latent_128",
    "20260122_latent_256",
    "20251110_latent_512",
    "20251110_latent_1024",
    "20260130_latent_512",
    "20251110_latent_1024",
]

# ============================
# Main Workflow Entry Point
# ============================

def main(
    model_manifest_name: list[str] = DEFAULT_MODEL_MANIFEST_NAMES,
    run_name: list[str] | None = DEFAULT_RUN_NAMES,
    random_seed: int = RANDOM_SEED,
    mode: str = "basic",
    include_negative_controls: bool = False,
    save_intermediate_plots: bool = False,
    save_crops_as_tiff: bool = False,
    compute_baseline: bool = True,
    num_seeds: int = 1,
) -> None:
    """
    Run quality check assessment for trained Diffusion Autoencoder models.

    This unified workflow supports two main modes:
    - "basic": Single model evaluation with visual QC and optional negative controls
    - "comparison": Multi-model evaluation with quantitative metrics and comparison plots

    Parameters
    ----------
    model_manifest_name
        Name(s) of the model manifest to load. Can be a single string for basic mode,
        or a list for comparison mode. Defaults to the 10-model latent dimension 
        comparison study (8 BF latent dims 8-1024 + 2 CDH5 conditioned 512, 1024).
        For single model: --model_manifest_name model1
        For multiple models: --model_manifest_name model1 --model_manifest_name model2
    run_name
        Run name(s) within the model manifest to load. Defaults to the runs for the
        10-model latent comparison study. Provide once per model when using multiple manifests.
    random_seed
        Random seed for reproducibility of noise generation. When num_seeds > 1,
        this becomes the center of the seed range.
    mode
        Workflow mode:
        - "basic": Single model visual QC (original model_qc behavior)
        - "comparison": Multi-model comparison with metrics
    include_negative_controls
        If True, generates denoising with scrambled embeddings and scrambled input
        images as negative controls. Default False for comparison mode.
    save_intermediate_plots
        If True, saves individual contact sheets for each example. Default False.
    save_crops_as_tiff
        If True, saves crops as individual TIFF files. Default True. Only saved 
        for default seed.
    compute_baseline
        If True, computes baseline metrics by comparing ground truth CDH5 to CDH5
        from the next timepoint (timepoint + 1). This provides a temporal baseline
        for evaluating model performance. Only used in comparison mode. Default True.
    num_seeds
        Number of random seeds to evaluate. If > 1, evaluates across multiple seeds
        centered around random_seed and averages the results.

    Examples
    --------
    Default: 10-model latent dimension comparison study:
        endopipe model_qc --mode comparison

    Basic QC for a single model:
        endopipe model_qc --model_manifest_name my_model --run_name my_run

    Basic QC with negative controls:
        endopipe model_qc --model_manifest_name my_model \\
            --run_name my_run --include_negative_controls

    Compare multiple models with metrics:
        endopipe model_qc --mode comparison \\
            --model_manifest_name model1 --model_manifest_name model2 \\
            --run_name run1 --run_name run2

    Full analysis with all features (default 10 models):
        endopipe model_qc --mode comparison \\
            --include_negative_controls --save_intermediate_plots --num_seeds 10
    """
    import logging
    from typing import Any

    from endo_pipeline.cli import DEMO_MODE
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import evaluate_single_model
    from endo_pipeline.library.visualize.model_qc_plots import (
        LATENT_COMPARISON_LABELS,
        create_comparison_bar_plot,
    )
    from endo_pipeline.settings.examples import (
        MODEL_QC_EXAMPLES_REP_2_POSITIONS,
        MODEL_QC_EXAMPLES_TRAINING_POSITIONS,
        MODEL_QC_EXAMPLES_VALIDATION_POSITIONS,
    )

    logger = logging.getLogger(__name__)

    # Normalize inputs to lists
    if isinstance(model_manifest_name, str):
        model_manifest_names = [model_manifest_name]
    else:
        model_manifest_names = list(model_manifest_name)

    if run_name is None:
        run_names = [None] * len(model_manifest_names)
    elif isinstance(run_name, str):
        run_names = [run_name]
    else:
        run_names = list(run_name)

    # Ensure we have matching numbers of manifests and run names
    if len(run_names) != len(model_manifest_names):
        raise ValueError(
            f"Number of run_names ({len(run_names)}) must match "
            f"number of model_manifest_names ({len(model_manifest_names)})"
        )

    # Determine mode-based settings
    is_comparison_mode = mode == "comparison"
    compute_metrics = is_comparison_mode
    is_multi_model = len(model_manifest_names) > 1

    # For basic mode with single model, default to include negative controls if not set
    if not is_comparison_mode and not is_multi_model:
        include_negative_controls = True  # Default behavior for basic QC

    logger.info(f"Running model_qc in '{mode}' mode")
    logger.info(f"Models: {len(model_manifest_names)}, Compute metrics: {compute_metrics}")
    logger.info(f"Negative controls: {include_negative_controls}")

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
    example_sets_for_metrics = {"validation_positions", "rep_2_positions"}

    if DEMO_MODE:
        logger.info("DEMO MODE: Limiting MODEL_QC_EXAMPLES to training set")
        example_sets_all = [example_sets_all[0]]
        example_sets_for_metrics = {"training_positions"}

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

    # Storage for all results across seeds
    all_seed_results: dict[int, dict[int, dict]] = {
        model_idx: {} for model_idx in range(len(model_manifest_names))
    }

    # Evaluation wrapper
    def evaluate_model_seed(model_idx, manifest_name, run_name_input, seed):
        """Wrapper function for parallel evaluation."""
        is_default = seed == random_seed
        return evaluate_single_model(
            model_idx=model_idx,
            manifest_name=manifest_name,
            run_name_input=run_name_input,
            random_seed=seed,
            example_sets_all=example_sets_all,
            example_sets_for_metrics=example_sets_for_metrics,
            save_intermediate_plots=save_intermediate_plots,
            save_crops_as_tiff=save_crops_as_tiff,
            include_negative_controls=include_negative_controls,
            compute_metrics=compute_metrics,
            compute_baseline=compute_baseline and is_comparison_mode,
            is_default_seed=is_default,
        )

    logger.info("Running model evaluations...")
    for model_idx, (manifest_name, run_name_input) in enumerate(
        zip(model_manifest_names, run_names, strict=False)
    ):
        for seed in seeds_to_evaluate:
            result = evaluate_model_seed(model_idx, manifest_name, run_name_input, seed)
            all_seed_results[model_idx][seed] = result

    # Create comparison plots only for comparison mode
    if compute_metrics:
        logger.info("Aggregating metrics across seeds...")

        # Structure for final aggregated data
        all_metrics: dict[str, list[dict]] = {"validation_positions": [], "rep_2_positions": []}
        model_labels = []

        for model_idx in range(len(model_manifest_names)):
            seed_results = all_seed_results[model_idx]

            # Get model label from any seed result
            first_result = next(iter(seed_results.values()))
            model_labels.append(first_result["model_label"])

            for example_set_label in example_sets_for_metrics:
                # Collect all metrics across seeds
                all_corrs = []
                all_ssims = []
                all_lpips = []
                all_baseline_corrs = []
                all_baseline_ssims = []
                all_baseline_lpips = []

                for seed, result in seed_results.items():
                    all_corrs.extend(result[example_set_label]["correlations_100"])
                    all_ssims.extend(result[example_set_label]["ssims_100"])
                    all_lpips.extend(result[example_set_label]["lpips_100"])
                    # Collect baseline metrics if available
                    if "baseline_correlations" in result[example_set_label]:
                        all_baseline_corrs.extend(result[example_set_label]["baseline_correlations"])
                        all_baseline_ssims.extend(result[example_set_label]["baseline_ssims"])
                        all_baseline_lpips.extend(result[example_set_label]["baseline_lpips"])

                aggregated_metrics = {
                    "model_idx": model_idx,
                    "model_label": first_result["model_label"],
                    "example_set": example_set_label,
                    "correlations_100": all_corrs,
                    "ssims_100": all_ssims,
                    "lpips_100": all_lpips,
                    "baseline_correlations": all_baseline_corrs,
                    "baseline_ssims": all_baseline_ssims,
                    "baseline_lpips": all_baseline_lpips,
                    "num_seeds": len(seeds_to_evaluate),
                }
                all_metrics[example_set_label].append(aggregated_metrics)

        # Create comparison plots across all models
        logger.info("Creating comparison plots across models...")

        seed_suffix = f"_seeds_{len(seeds_to_evaluate)}" if len(seeds_to_evaluate) > 1 else ""
        comparison_output_path = get_output_path(
            "model_qc",
            "comparison",
            f"models_{len(model_manifest_names)}{seed_suffix}",
        )

        # Prepare data for plotting
        models_data: list[dict[str, Any]] = []
        
        # Collect baseline metrics (same across all models since it's temporal comparison)
        baseline_data: dict[str, dict[str, float]] = {
            "validation": {"corr_mean": 0.0, "corr_std": 0.0, "ssim_mean": 0.0, "ssim_std": 0.0, "lpips_mean": 0.0, "lpips_std": 0.0},
            "rep2": {"corr_mean": 0.0, "corr_std": 0.0, "ssim_mean": 0.0, "ssim_std": 0.0, "lpips_mean": 0.0, "lpips_std": 0.0},
        }
        
        # Aggregate baseline from first model (it's the same for all since it's independent of model)
        if compute_baseline:
            all_baseline_corrs_val = []
            all_baseline_ssims_val = []
            all_baseline_lpips_val = []
            all_baseline_corrs_rep2 = []
            all_baseline_ssims_rep2 = []
            all_baseline_lpips_rep2 = []
            
            for data in all_metrics["validation_positions"]:
                if data["baseline_correlations"]:
                    all_baseline_corrs_val.extend(data["baseline_correlations"])
                    all_baseline_ssims_val.extend(data["baseline_ssims"])
                    all_baseline_lpips_val.extend(data["baseline_lpips"])
                    break  # Baseline is same for all models
            
            for data in all_metrics["rep_2_positions"]:
                if data["baseline_correlations"]:
                    all_baseline_corrs_rep2.extend(data["baseline_correlations"])
                    all_baseline_ssims_rep2.extend(data["baseline_ssims"])
                    all_baseline_lpips_rep2.extend(data["baseline_lpips"])
                    break
            
            if all_baseline_corrs_val:
                baseline_data["validation"] = {
                    "corr_mean": np.mean(all_baseline_corrs_val),
                    "corr_std": np.std(all_baseline_corrs_val),
                    "ssim_mean": np.mean(all_baseline_ssims_val),
                    "ssim_std": np.std(all_baseline_ssims_val),
                    "lpips_mean": np.mean(all_baseline_lpips_val),
                    "lpips_std": np.std(all_baseline_lpips_val),
                }
            if all_baseline_corrs_rep2:
                baseline_data["rep2"] = {
                    "corr_mean": np.mean(all_baseline_corrs_rep2),
                    "corr_std": np.std(all_baseline_corrs_rep2),
                    "ssim_mean": np.mean(all_baseline_ssims_rep2),
                    "ssim_std": np.std(all_baseline_ssims_rep2),
                    "lpips_mean": np.mean(all_baseline_lpips_rep2),
                    "lpips_std": np.std(all_baseline_lpips_rep2),
                }
        
        for model_idx in range(len(model_manifest_names)):
            model_entry: dict[str, Any] = {
                "model_idx": model_idx,
                "model_label": None,
                "validation": {
                    "corr_mean": 0.0,
                    "corr_std": 0.0,
                    "ssim_mean": 0.0,
                    "ssim_std": 0.0,
                    "lpips_mean": 0.0,
                    "lpips_std": 0.0,
                },
                "rep2": {
                    "corr_mean": 0.0,
                    "corr_std": 0.0,
                    "ssim_mean": 0.0,
                    "ssim_std": 0.0,
                    "lpips_mean": 0.0,
                    "lpips_std": 0.0,
                },
                "baseline_validation": baseline_data["validation"] if compute_baseline else None,
                "baseline_rep2": baseline_data["rep2"] if compute_baseline else None,
            }
            # Find validation data for this model
            for data in all_metrics["validation_positions"]:
                if data["model_idx"] == model_idx:
                    model_entry["model_label"] = data["model_label"]
                    model_entry["validation"]["corr_mean"] = np.mean(data["correlations_100"])
                    model_entry["validation"]["corr_std"] = np.std(data["correlations_100"])
                    model_entry["validation"]["ssim_mean"] = np.mean(data["ssims_100"])
                    model_entry["validation"]["ssim_std"] = np.std(data["ssims_100"])
                    model_entry["validation"]["lpips_mean"] = np.mean(data["lpips_100"])
                    model_entry["validation"]["lpips_std"] = np.std(data["lpips_100"])
                    break
            # Find rep2 data for this model
            for data in all_metrics["rep_2_positions"]:
                if data["model_idx"] == model_idx:
                    model_entry["rep2"]["corr_mean"] = np.mean(data["correlations_100"])
                    model_entry["rep2"]["corr_std"] = np.std(data["correlations_100"])
                    model_entry["rep2"]["ssim_mean"] = np.mean(data["ssims_100"])
                    model_entry["rep2"]["ssim_std"] = np.std(data["ssims_100"])
                    model_entry["rep2"]["lpips_mean"] = np.mean(data["lpips_100"])
                    model_entry["rep2"]["lpips_std"] = np.std(data["lpips_100"])
                    break
            models_data.append(model_entry)

        # Determine model labels:
        # Use latent comparison labels only when we have exactly 10 models
        # (matching the expected latent dimension comparison setup)
        # Otherwise use generic "Model 1", "Model 2", etc.
        if len(models_data) == len(LATENT_COMPARISON_LABELS):
            model_labels = LATENT_COMPARISON_LABELS
        else:
            model_labels = [f"Model {i+1}" for i in range(len(models_data))]

        seeds_info = (
            f" (averaged over {len(seeds_to_evaluate)} seeds)" if len(seeds_to_evaluate) > 1 else ""
        )
        legend_text = f"Model Details{seeds_info}:\n" + "\n".join(
            [
                f"{model_labels[i]}: "
                f"{model_manifest_names[i]}\n         Run: {run_names[i] if run_names[i] else 'latest'}"
                for i in range(len(models_data))
            ]
        )

        # Create comparison plots
        create_comparison_bar_plot(
            models_data=models_data,
            metric_key="corr",
            ylabel="Pearson Correlation (100% Noise)",
            title=f"Correlation{seeds_info}",
            output_path=comparison_output_path,
            filename="correlation_comparison_100_noise",
            legend_text=legend_text,
            model_labels=model_labels,
            ylim=(0, 1.0),
            show_baseline=compute_baseline,
        )
        create_comparison_bar_plot(
            models_data=models_data,
            metric_key="ssim",
            ylabel="SSIM Score (100% Noise)",
            title=f"SSIM{seeds_info}",
            output_path=comparison_output_path,
            filename="ssim_comparison_100_noise",
            legend_text=legend_text,
            model_labels=model_labels,
            ylim=(0, 1.0),
            show_baseline=compute_baseline,
        )
        create_comparison_bar_plot(
            models_data=models_data,
            metric_key="lpips",
            ylabel="LPIPS Score (100% Noise)",
            title=f"LPIPS{seeds_info}",
            output_path=comparison_output_path,
            filename="lpips_comparison_100_noise",
            legend_text=legend_text,
            model_labels=model_labels,
            text_box_loc="lower right",
            show_baseline=compute_baseline,
        )

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info(f"SUMMARY: Model Performance{seeds_info}")
        logger.info("=" * 80)
        
        # Print baseline summary if computed
        if compute_baseline and baseline_data["validation"]["corr_mean"] > 0:
            logger.info("\nBASELINE (Temporal - Next Timepoint Comparison):")
            logger.info(
                f"  Validation - Corr: {baseline_data['validation']['corr_mean']:.3f} ± "
                f"{baseline_data['validation']['corr_std']:.3f}, "
                f"SSIM: {baseline_data['validation']['ssim_mean']:.3f} ± "
                f"{baseline_data['validation']['ssim_std']:.3f}, "
                f"LPIPS: {baseline_data['validation']['lpips_mean']:.3f} ± "
                f"{baseline_data['validation']['lpips_std']:.3f}"
            )
            logger.info(
                f"  Rep2       - Corr: {baseline_data['rep2']['corr_mean']:.3f} ± "
                f"{baseline_data['rep2']['corr_std']:.3f}, "
                f"SSIM: {baseline_data['rep2']['ssim_mean']:.3f} ± "
                f"{baseline_data['rep2']['ssim_std']:.3f}, "
                f"LPIPS: {baseline_data['rep2']['lpips_mean']:.3f} ± "
                f"{baseline_data['rep2']['lpips_std']:.3f}"
            )
            logger.info("-" * 80)
        
        for model_data in models_data:
            logger.info(f"\n{model_data['model_label']}:")
            logger.info(
                f"  Validation - Corr: {model_data['validation']['corr_mean']:.3f} ± "
                f"{model_data['validation']['corr_std']:.3f}, "
                f"SSIM: {model_data['validation']['ssim_mean']:.3f} ± "
                f"{model_data['validation']['ssim_std']:.3f}, "
                f"LPIPS: {model_data['validation']['lpips_mean']:.3f} ± "
                f"{model_data['validation']['lpips_std']:.3f}"
            )
            logger.info(
                f"  Rep2       - Corr: {model_data['rep2']['corr_mean']:.3f} ± "
                f"{model_data['rep2']['corr_std']:.3f}, "
                f"SSIM: {model_data['rep2']['ssim_mean']:.3f} ± "
                f"{model_data['rep2']['ssim_std']:.3f}, "
                f"LPIPS: {model_data['rep2']['lpips_mean']:.3f} ± "
                f"{model_data['rep2']['lpips_std']:.3f}"
            )
        logger.info("=" * 80)

    logger.info("Model QC workflow completed successfully!")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
