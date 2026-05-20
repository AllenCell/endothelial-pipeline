"""Production workflow: persist Model-QC inference metrics for the supp. figure.

Runs diffusion-autoencoder denoising for the curated 10-model latent-dim
sweep on the Rep-2 example positions, computes per-example correlation /
SSIM / LPIPS plus a next-timepoint baseline, and persists everything as
a :class:`DataframeManifest` of per-model parquets so the companion
``fig-model-qc-plot`` workflow can render the bar chart without re-running
GPU inference.

This is a publication-figure driver: model identities, example set, seed
range, and noise levels are all hard-coded.  Use the ``model-qc``
development workflow for ad-hoc multi-model QC.
"""


def main(resume: bool = False) -> None:
    """Run inference and emit a dataframe manifest of per-model parquets.

    #diffae #figure #gpu #production

    Whether per-model parquets are uploaded to FMS (and to which
    environment) is governed by the global ``--upload-to-fms`` /
    ``--fms-environment`` CLI flags, which the entry-point app maps onto
    the ``UPLOAD_TO_FMS`` / ``FMS_ENVIRONMENT`` module constants in
    :mod:`endo_pipeline.cli`.

    Parameters
    ----------
    resume
        Skip ``(model, seed)`` pairs whose shard parquet already exists
        under ``<outdir>/shards/``.  Lets a partially-completed sweep
        be restarted without re-burning GPU time on pairs already done.
        Default: ``False``.
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS, UPLOAD_TO_FMS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model.model_qc import (
        ModelKey,
        evaluate_single_model,
    )
    from endo_pipeline.library.model.model_qc.results_io import (
        model_key_str,
        save_shard,
        shard_path,
        write_model_parquet_from_shards,
    )
    from endo_pipeline.manifests import (
        DataframeLocation,
        build_dataframe_location_from_path,
        create_dataframe_manifest,
        get_most_recent_run_name,
        load_model_manifest,
        save_dataframe_manifest,
    )
    from endo_pipeline.settings.examples import MODEL_QC_EXAMPLES_REP_2_POSITIONS
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_NAME,
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

    # Seed expansion mirrors development/model_qc.py.
    half_range = num_seeds // 2
    seeds_to_evaluate = list(range(RANDOM_SEED - half_range, RANDOM_SEED - half_range + num_seeds))
    logger.info("Seeds: %s (default %d)", seeds_to_evaluate, RANDOM_SEED)

    # Resolve ModelKeys eagerly so the (manifest, run) pair is a stable
    # key across output paths, dataframe-manifest keys, and downstream
    # plot labels.
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

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first 2 models and 2 seeds.")
        model_keys = model_keys[:2]
        seeds_to_evaluate = seeds_to_evaluate[:2]

    metrics_dir = get_output_path(__file__, "metrics")
    sample_images_dir = get_output_path(__file__, "sample_images", create_directories=True)
    logger.info("Persisting per-model metrics parquets to: %s", metrics_dir)
    logger.info("Persisting sample-image crops to:        %s", sample_images_dir)

    # Initialize / load the output dataframe manifest up front so it
    # exists on disk even if the run is interrupted before the first
    # model finishes.
    demo_suffix = "_demo" if DEMO_MODE else ""
    out_manifest_name = f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_NAME}{demo_suffix}"
    out_manifest = create_dataframe_manifest(out_manifest_name, workflow_name=__file__)
    out_manifest.parameters = {
        "num_seeds": num_seeds,
        "seeds": [int(s) for s in seeds_to_evaluate],
        "random_seed": RANDOM_SEED,
        "example_set": example_set_label,
        "noise_levels": list(MODEL_QC_NOISE_LEVELS),
        "manifest_names": list(DEFAULT_MODEL_QC_MANIFEST_NAMES),
        "run_names": list(DEFAULT_MODEL_QC_RUN_NAMES),
    }
    save_dataframe_manifest(out_manifest)

    # --- Per-model inference loop ---
    for model_key in model_keys:
        for seed in seeds_to_evaluate:
            target = shard_path(metrics_dir, model_key, seed)
            if resume and target.exists():
                logger.info(
                    "[resume] Skipping %s seed=%d (shard exists: %s)",
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
            save_shard(
                metrics_dir,
                model_key,
                seed,
                result,
                examples_by_set={example_set_label: examples},
            )

        # All shards for this model are on disk -- consolidate into the
        # per-model parquet and add to the dataframe manifest.
        model_parquet = write_model_parquet_from_shards(metrics_dir, model_key, seeds_to_evaluate)

        location_key = model_key_str(model_key)
        if UPLOAD_TO_FMS:
            unique_dataset_names = sorted({e.dataset_name for e in examples})
            dataset_configs = [load_dataset_config(n) for n in unique_dataset_names]
            notes = (
                f"Model-QC supplementary figure metrics for "
                f"{model_key.manifest_name}/{model_key.run_name}: "
                f"{len(seeds_to_evaluate)} seeds x {len(examples)} crops, "
                "long-format (one row per seed x crop)."
            )
            annotations = build_fms_annotations(dataset_configs, additional_notes=notes)
            fmsid = upload_file_to_fms(model_parquet, annotations=annotations, file_type="parquet")
            out_manifest.locations[location_key] = DataframeLocation(fmsid=fmsid)
            logger.info("Uploaded %s to FMS as %s", model_parquet.name, fmsid)
        elif (
            out_manifest.locations.get(location_key) is None
            or out_manifest.locations[location_key].fmsid is None
        ):
            out_manifest.locations[location_key] = build_dataframe_location_from_path(model_parquet)

        save_dataframe_manifest(out_manifest)

    logger.info("Inference complete. Run `endopipe fig-model-qc-plot` to render the figure.")


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
