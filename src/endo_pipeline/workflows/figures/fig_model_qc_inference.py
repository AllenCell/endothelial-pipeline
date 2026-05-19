"""Inference half of the supplementary Model-QC figure pipeline.

Runs diffusion-autoencoder denoising for the curated 10-model latent-dim
sweep on the Rep-2 example positions, computes per-example correlation /
SSIM / LPIPS plus a next-timepoint baseline, and persists everything to
disk so the companion plot workflow can render the bar chart without
re-running GPU inference.

This is a publication-figure driver: model identities, example set, seed
range, and noise levels are all hard-coded.  Use ``model-qc`` (under
development workflows) for ad-hoc multi-model QC.
"""

import logging


def main(resume: bool = False) -> None:
    """Run diffusion inference and persist per-(model, seed) JSON results.

    #diffae #figure #gpu

    Parameters
    ----------
    resume
        Skip ``(model, seed)`` pairs whose result JSON already exists in the
        target output directory.  Useful for restarting a partially completed
        sweep without re-burning GPU time.  Default: ``False``.
    """
    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.model.model_qc import (
        ModelKey,
        evaluate_single_model,
        save_seed_result,
        seed_result_path,
        write_combined_dataframe,
        write_inference_manifest,
    )
    from endo_pipeline.manifests import get_most_recent_run_name, load_model_manifest
    from endo_pipeline.settings.examples import MODEL_QC_EXAMPLES_REP_2_POSITIONS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
        MODEL_QC_NOISE_LEVELS,
        RANDOM_SEED,
    )

    logger = logging.getLogger(__name__)

    # --- Hard-coded panel configuration ---
    num_seeds = 10
    example_set_label = "rep_2_positions"
    example_sets_for_metrics = {example_set_label}

    examples = MODEL_QC_EXAMPLES_REP_2_POSITIONS
    if DEMO_MODE:
        logger.info("DEMO MODE: limiting Rep 2 examples to first entry")
        examples = examples[:1]
    example_sets_all = [(examples, example_set_label)]

    # Seed expansion mirrors model_qc.py's logic so default-seed crops match.
    half_range = num_seeds // 2
    seeds_to_evaluate = list(range(RANDOM_SEED - half_range, RANDOM_SEED - half_range + num_seeds))
    logger.info("Seeds: %s (default %d)", seeds_to_evaluate, RANDOM_SEED)

    # Resolve ModelKeys eagerly (matches model_qc.py); reject duplicates.
    model_keys: list[ModelKey] = []
    for manifest_name, run_name in zip(
        DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True
    ):
        if run_name is None:
            run_name = get_most_recent_run_name(load_model_manifest(manifest_name))
        model_keys.append(ModelKey(manifest_name, run_name))
    duplicates = [k for k in model_keys if model_keys.count(k) > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate (manifest_name, run_name) entries are not allowed: "
            f"{sorted(set(duplicates))}"
        )

    run_dir = get_output_path("model_qc_supp", "metrics")
    sample_images_dir = get_output_path(
        "model_qc_supp", "sample_images", create_directories=True
    )
    logger.info("Persisting inference metrics to: %s", run_dir)
    logger.info("Persisting sample-image crops to: %s", sample_images_dir)

    # Write manifest up-front so partial runs are still discoverable by the
    # plot workflow (which reads it to know which (model, seed) pairs to
    # look for).
    write_inference_manifest(
        run_dir, model_keys, seeds_to_evaluate, example_set_labels=[example_set_label]
    )

    for model_key in model_keys:
        for seed in seeds_to_evaluate:
            target = seed_result_path(run_dir, model_key, seed)
            if resume and target.exists():
                logger.info(
                    "[resume] Skipping %s seed=%d (exists: %s)",
                    model_key.label.replace("\n", " / "),
                    seed,
                    target.name,
                )
                continue

            is_default = seed == RANDOM_SEED
            result = evaluate_single_model(
                model_key=model_key,
                random_seed=seed,
                example_sets_all=example_sets_all,
                example_sets_for_metrics=example_sets_for_metrics,
                save_intermediate_plots=False,
                save_crops_as_tiff=is_default,
                include_negative_controls=False,
                compute_metrics=True,
                noise_levels=MODEL_QC_NOISE_LEVELS,
                compute_baseline=True,
                is_default_seed=is_default,
                num_gpus=NUM_GPUS,
                output_root=sample_images_dir,
            )
            save_seed_result(
                run_dir,
                model_key,
                seed,
                result,
                examples_by_set={example_set_label: examples},
            )

    combined_path = write_combined_dataframe(run_dir)
    logger.info("Wrote consolidated metrics dataframe: %s", combined_path)
    logger.info("Inference complete. Run `endopipe fig-model-qc-plot` to render the figure.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
