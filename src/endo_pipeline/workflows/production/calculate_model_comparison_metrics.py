def main() -> None:
    """
    Calculate comparison metrics between DiffAE models.

    #diffae #model-comparison #gpu

    This workflow compares DiffAE models trained with different semantic conditioning
    image types and number of latent dimensions by calculating per-example
    correlation, SSIM, and LPIPS metrics.

    ## Example usage

    To run the workflow in demo mode:

    ```bash
    uv run endopipe calculate-model-comparison-metrics -vd
    ```

    To run the full workflow:

    ```bash
    uv run endopipe calculate-model-comparison-metrics
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will calculate
    comparison metrics for only two models and two random seeds.
    """
    import logging

    from endo_pipeline.cli import DEMO_MODE, NUM_GPUS, UPLOAD_TO_FMS
    from endo_pipeline.configs import load_dataset_config
    from endo_pipeline.io import build_fms_annotations, get_output_path, upload_file_to_fms
    from endo_pipeline.library.model.model_qc import (
        ModelKey,
        evaluate_single_model,
        write_per_model_parquets,
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
        DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX,
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
    logger.info("Persisting per-model metrics parquets under: %s", metrics_dir)
    logger.info("Persisting sample-image crops to:           %s", sample_images_dir)

    # --- Per-(model, seed) inference loop ---
    results: list[tuple[ModelKey, int, dict]] = []
    for model_key in model_keys:
        for seed in seeds_to_evaluate:
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
            results.append((model_key, seed, result))

    # --- Persist per-(model, run) parquets ---
    parquet_paths = write_per_model_parquets(
        metrics_dir,
        results,
        examples_by_set={example_set_label: examples},
    )

    demo_suffix = "_demo" if DEMO_MODE else ""
    base_parameters = {
        "num_seeds": num_seeds,
        "seeds": [int(s) for s in seeds_to_evaluate],
        "random_seed": RANDOM_SEED,
        "example_set": example_set_label,
        "noise_levels": list(MODEL_QC_NOISE_LEVELS),
    }

    if UPLOAD_TO_FMS:
        unique_dataset_names = sorted({e.dataset_name for e in examples})
        dataset_configs = [load_dataset_config(n) for n in unique_dataset_names]

    # --- Emit one DataframeManifest per manifest_name ---
    # Group by manifest_name to mirror the per-model-manifest structure;
    # each output manifest's locations dict is keyed by run_name.
    keys_by_manifest: dict[str, list[ModelKey]] = {}
    for mk in model_keys:
        keys_by_manifest.setdefault(mk.manifest_name, []).append(mk)

    for manifest_name, keys in keys_by_manifest.items():
        out_manifest_name = (
            f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{manifest_name}{demo_suffix}"
        )
        out_manifest = create_dataframe_manifest(out_manifest_name, workflow_name=__file__)
        # Discard stale locations from any previous run so the rewritten
        # manifest only references the parquets just produced.
        out_manifest.locations.clear()
        out_manifest.parameters = {
            **base_parameters,
            "manifest_name": manifest_name,
            "run_names": [k.run_name for k in keys],
        }

        for mk in keys:
            parquet = parquet_paths[mk]
            if UPLOAD_TO_FMS:
                notes = (
                    f"Model-QC supplementary figure metrics for "
                    f"{mk.manifest_name}/{mk.run_name}: {len(seeds_to_evaluate)} seeds x "
                    f"{len(examples)} crops, long-format (one row per seed x crop)."
                )
                annotations = build_fms_annotations(dataset_configs, additional_notes=notes)
                fmsid = upload_file_to_fms(parquet, annotations=annotations, file_type="parquet")
                out_manifest.locations[mk.run_name] = DataframeLocation(fmsid=fmsid)
                logger.info("Uploaded %s to FMS as %s", parquet.name, fmsid)
            else:
                out_manifest.locations[mk.run_name] = build_dataframe_location_from_path(parquet)

        save_dataframe_manifest(out_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
