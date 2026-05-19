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


def main(resume: bool = False, upload_to_fms: bool = False) -> None:
    """Run diffusion inference and persist per-(model, seed) parquet results.

    #diffae #figure #gpu

    Parameters
    ----------
    resume
        Skip ``(model, seed)`` pairs whose result parquet already exists
        in the target output directory.  Lets a partially-completed sweep
        be restarted without re-burning GPU time on pairs already done
        (resume key = the parquet filename, which is fully determined by
        ``manifest_name``, ``run_name``, and ``seed``).  Default: ``False``.
    upload_to_fms
        If true, upload the consolidated ``model_qc_metrics.parquet`` to
        FMS after inference completes.  The FMS notes annotation enumerates
        the unique datasets touched by the example set so the asset is
        traceable.  Default: ``False`` (the parquet stays local).
    """
    from endo_pipeline.cli import NUM_GPUS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model.model_qc import (
        ModelKey,
        evaluate_single_model,
        save_seed_result,
        seed_result_path,
        write_combined_dataframe,
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
    sample_images_dir = get_output_path("model_qc_supp", "sample_images", create_directories=True)
    logger.info("Persisting inference metrics to: %s", run_dir)
    logger.info("Persisting sample-image crops to: %s", sample_images_dir)

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
                output_path=sample_images_dir,
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

    if upload_to_fms:
        unique_dataset_names = sorted({e.dataset_name for e in examples})
        dataset_configs = [load_dataset_config(n) for n in unique_dataset_names]
        notes = (
            f"Model-QC supplementary figure metrics: {len(model_keys)} models "
            f"x {len(seeds_to_evaluate)} seeds x {len(examples)} crops, "
            "long-format (one row per model x seed x crop)."
        )
        annotations = build_fms_annotations(dataset_configs, additional_notes=notes)
        fmsid = upload_file_to_fms(combined_path, annotations=annotations, file_type="parquet")
        logger.info("Uploaded %s to FMS as %s", combined_path.name, fmsid)

    logger.info("Inference complete. Run `endopipe fig-model-qc-plot` to render the figure.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
