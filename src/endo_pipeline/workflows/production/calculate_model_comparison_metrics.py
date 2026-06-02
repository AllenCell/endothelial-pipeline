def main(model_manifest_name: str, run_names: list[str] | None = None) -> None:
    r"""
    Calculate comparison metrics for one DiffAE model manifest.

    #diffae #model-comparison #gpu

    This workflow evaluates the runs of a single DiffAE model manifest by
    calculating per-example correlation, SSIM, and LPIPS metrics, and emits
    one ``DataframeManifest`` (with ``locations`` keyed by ``run_name``).

    ## Example usage

    Evaluate every run in a model manifest:

    ```bash
    uv run endopipe calculate-model-comparison-metrics diffae_baseline_exclude_cell_piling
    ```

    Evaluate a subset of runs:

    ```bash
    uv run endopipe calculate-model-comparison-metrics diffae_cdh5_conditioned \
        --run-names 20260130_latent_512 --run-names 20251110_latent_1024
    ```

    ## Workflow demo

    Running the workflow in demo mode (`-d` or `--demo-mode`) will calculate
    comparison metrics for only the first two runs in the model manifest and
    two random seeds.

    Parameters
    ----------
    model_manifest_name
        Name of the DiffAE model manifest to evaluate
        (e.g. ``"diffae_baseline_exclude_cell_piling"``).
    run_names
        Optional subset of ``run_name`` entries from the model manifest to
        evaluate. If ``None``, defaults to the curated QC subset from
        :data:`DEFAULT_MODEL_QC_RUN_NAMES` for this manifest (or all
        ``locations`` if the manifest is not in the curated QC list).
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
    model_manifest = load_model_manifest(model_manifest_name)
    if run_names is None:
        # Default to the curated QC run subset for this manifest_name. Model
        # manifests can contain legacy/non-QC runs (e.g. older architectures)
        # whose checkpoints would crash evaluate_single_model if loaded
        # blindly, so we only fall back to all locations when this manifest
        # is not in the curated QC list.
        curated = [
            r
            for m, r in zip(
                DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True
            )
            if m == model_manifest_name
        ]
        run_names = curated if curated else list(model_manifest.locations.keys())
    else:
        missing = [r for r in run_names if r not in model_manifest.locations]
        if missing:
            raise ValueError(
                f"run_names not present in model manifest {model_manifest_name!r}: {missing}"
            )
    if not run_names:
        raise ValueError(f"No runs to evaluate for model manifest {model_manifest_name!r}.")
    model_keys: list[ModelKey] = [ModelKey(model_manifest_name, r) for r in run_names]

    if DEMO_MODE:
        logger.warning("DEMO MODE: limiting to first 2 models and 2 seeds.")
        model_keys = model_keys[:2]
        seeds_to_evaluate = seeds_to_evaluate[:2]

    metrics_dir = get_output_path(__file__, "metrics")
    sample_images_dir = get_output_path(__file__, "sample_images", create_directories=True)
    logger.info("Persisting per-model metrics parquets under: %s", metrics_dir)
    logger.info("Persisting sample-image crops to:           %s", sample_images_dir)

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

    # --- Prepare the output DataframeManifest ---
    # locations are keyed by run_name. We intentionally do NOT clear
    # existing locations so partial reruns (subset of run_names) merge
    # into the existing manifest instead of wiping unrelated entries.
    out_manifest_name = (
        f"{DEFAULT_MODEL_QC_DATAFRAME_MANIFEST_PREFIX}_{model_manifest_name}{demo_suffix}"
    )
    out_manifest = create_dataframe_manifest(out_manifest_name, workflow_name=__file__)
    out_manifest.parameters = {**base_parameters, "manifest_name": model_manifest_name}

    # --- Per-run loop: evaluate seeds, persist parquet, update manifest ---
    # Saving the manifest after each run means a mid-sweep crash still
    # leaves prior runs catalogued for downstream consumers.
    for model_key in model_keys:
        results: list[tuple[ModelKey, int, dict]] = []
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

        parquet_paths = write_per_model_parquets(
            metrics_dir,
            results,
            examples_by_set={example_set_label: examples},
        )
        parquet = parquet_paths[model_key]

        if UPLOAD_TO_FMS:
            notes = (
                f"Model-QC supplementary figure metrics for "
                f"{model_key.manifest_name}/{model_key.run_name}: "
                f"{len(seeds_to_evaluate)} seeds x {len(examples)} crops, "
                f"long-format (one row per seed x crop)."
            )
            annotations = build_fms_annotations(dataset_configs, additional_notes=notes)
            fmsid = upload_file_to_fms(parquet, annotations=annotations, file_type="parquet")
            out_manifest.locations[model_key.run_name] = DataframeLocation(fmsid=fmsid)
            logger.info("Uploaded %s to FMS as %s", parquet.name, fmsid)
        else:
            out_manifest.locations[model_key.run_name] = build_dataframe_location_from_path(parquet)

        save_dataframe_manifest(out_manifest)


if __name__ == "__main__":
    from endo_pipeline.cli import workflow_cli

    workflow_cli(main)
