"""Metric computation, aggregation, and comparison plot generation for model QC."""

from typing import TYPE_CHECKING, Any

import numpy as np

from endo_pipeline.library.process.image_processing import crop_image
from endo_pipeline.library.visualize.model_inputs.image_preprocessing_steps import (
    get_target_image_from_sample,
)

from .image_loading import load_transformed_image
from .image_metrics import compute_correlation, compute_ssim

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from endo_pipeline.settings.examples import ExampleImage

    from .evaluation import ModelKey
    from .image_metrics import LPIPSCalculator


# ---------------------------------------------------------------------------
# Baseline per-example metric
# ---------------------------------------------------------------------------


def compute_baseline_for_example(
    example: "ExampleImage",
    model_config: "DictConfig",
    diffusion_input_key: str,
    crop_size: int,
    ground_truth: np.ndarray,
    lpips_calculator: "LPIPSCalculator",
) -> tuple[float, float, float]:
    """Compare ground truth to the next-timepoint CDH5 crop.

    Parameters
    ----------
    example
        The example image metadata.
    model_config
        The model configuration loaded from MLflow.
    diffusion_input_key
        Key for the diffusion input channel.
    crop_size
        Size of the square crop in pixels.
    ground_truth
        Squeezed ground-truth diffusion image.
    lpips_calculator
        Pre-initialised LPIPS metric calculator.

    Returns
    -------
    Metrics in the format of
        ``(pearson_correlation, ssim, lpips)``.

    Raises
    ------
    Exception
        If the next-timepoint image cannot be loaded or metrics cannot be
        computed (e.g. missing timepoint). Callers should handle this.
    """
    sample_next = load_transformed_image(
        example,
        model_config,
        timepoint=example.timepoint + 1,
    )
    diffusion_next = get_target_image_from_sample(
        sample_next,
        target_key=diffusion_input_key,
    )
    crop_next = crop_image(
        diffusion_next,
        example.crop_x_start,
        example.crop_y_start,
        crop_size,
    ).squeeze()

    corr = compute_correlation(ground_truth, crop_next)
    ssim_val = compute_ssim(
        ground_truth,
        crop_next,
        data_range=2.0,
    )
    lpips_val = lpips_calculator.compute(ground_truth, crop_next)
    return corr, ssim_val, lpips_val


# ---------------------------------------------------------------------------
# Seed-level aggregation
# ---------------------------------------------------------------------------


def aggregate_seed_metrics(
    all_seed_results: dict["ModelKey", dict[int, dict]],
    model_keys: list["ModelKey"],
    example_sets_for_metrics: set[str],
    seeds_to_evaluate: list[int],
) -> tuple[dict[str, list[dict]], list[str]]:
    """Aggregate per-seed metrics into combined metric dictionaries per model.

    Flattens the per-seed, per-example correlation / SSIM / LPIPS lists into
    a single aggregated dictionary per model.

    Parameters
    ----------
    all_seed_results
        Nested mapping ``{model_key: {seed: result_dict}}``.  Each
        ``model_key`` is a ``ModelKey``.  Each ``result_dict`` contains
        per-example-set metric lists keyed by ``"correlations_100"``,
        ``"ssims_100"``, ``"lpips_100"``, ``"baseline_correlations"``,
        ``"baseline_ssims"``, and ``"baseline_lpips"`` (the last 3 are
        when baselines are computed).
    model_keys
        Ordered list of ``ModelKey``.  The iteration order determines the
        order of entries in the returned per-example-set lists (and
        therefore the bar-plot ordering).
    example_sets_for_metrics
        Example-set labels to aggregate
        (e.g. ``{"validation_positions", "rep_2_positions"}``).
    seeds_to_evaluate
        Seeds that were evaluated; stored in the output for provenance.

    Returns
    -------
    all_metrics
        Each aggregated dict has flat lists of all per-example, per-seed metric
        values plus metadata (``model_key``, ``model_label``, ``num_seeds``).
    model_labels
        Ordered human-readable model labels (one per model).
    """
    all_metrics: dict[str, list[dict]] = {label: [] for label in example_sets_for_metrics}
    model_labels: list[str] = []

    for model_key in model_keys:
        seed_results = all_seed_results[model_key]
        model_labels.append(model_key.label)

        for example_set_label in example_sets_for_metrics:
            all_corrs: list[float] = []
            all_ssims: list[float] = []
            all_lpips: list[float] = []
            all_baseline_corrs: list[float] = []
            all_baseline_ssims: list[float] = []
            all_baseline_lpips: list[float] = []

            for _seed, result in seed_results.items():
                all_corrs.extend(result[example_set_label]["correlations_100"])
                all_ssims.extend(result[example_set_label]["ssims_100"])
                all_lpips.extend(result[example_set_label]["lpips_100"])
                if "baseline_correlations" in result[example_set_label]:
                    all_baseline_corrs.extend(result[example_set_label]["baseline_correlations"])
                    all_baseline_ssims.extend(result[example_set_label]["baseline_ssims"])
                    all_baseline_lpips.extend(result[example_set_label]["baseline_lpips"])

            aggregated_metrics = {
                "model_key": model_key,
                "model_label": model_key.label,
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

    return all_metrics, model_labels


# ---------------------------------------------------------------------------
# Baseline data extraction
# ---------------------------------------------------------------------------


def compute_baseline_data(
    all_metrics: dict[str, list[dict]],
    compute_baseline: bool,
) -> dict[str, dict[str, float]]:
    """Extract baseline (temporal next-timepoint) statistics from aggregated metrics.

    The baseline compares each ground-truth crop to the CDH5 image at the
    next timepoint, giving an upper bound on how well a generative model
    should perform.

    Parameters
    ----------
    all_metrics
        Aggregated per-model metric dictionaries keyed by example-set label.
        Each dict must contain ``"baseline_correlations"``,
        ``"baseline_ssims"``, and ``"baseline_lpips"`` lists.
    compute_baseline
        If ``False``, returns zeroed-out baseline data immediately.

    Returns
    -------
    baseline_data
        Mapping with keys ``"validation"`` and ``"rep2"``, each containing:

        - ``"corr_mean"`` / ``"corr_std"`` — Pearson correlation mean and std.
        - ``"ssim_mean"`` / ``"ssim_std"`` — SSIM mean and std.
        - ``"lpips_mean"`` / ``"lpips_std"`` — LPIPS mean and std.

        All values are ``0.0`` when ``compute_baseline`` is ``False`` or no
        baseline data was available.
    """
    _zero: dict[str, float] = {
        "corr_mean": 0.0,
        "corr_std": 0.0,
        "ssim_mean": 0.0,
        "ssim_std": 0.0,
        "lpips_mean": 0.0,
        "lpips_std": 0.0,
    }
    baseline_data: dict[str, dict[str, float]] = {
        "validation": dict(_zero),
        "rep2": dict(_zero),
    }
    if not compute_baseline:
        return baseline_data

    for split_key, metric_key in [
        ("validation", "validation_positions"),
        ("rep2", "rep_2_positions"),
    ]:
        for data in all_metrics[metric_key]:
            if data["baseline_correlations"]:
                baseline_data[split_key] = {
                    "corr_mean": float(np.mean(data["baseline_correlations"])),
                    "corr_std": float(np.std(data["baseline_correlations"])),
                    "ssim_mean": float(np.mean(data["baseline_ssims"])),
                    "ssim_std": float(np.std(data["baseline_ssims"])),
                    "lpips_mean": float(np.mean(data["baseline_lpips"])),
                    "lpips_std": float(np.std(data["baseline_lpips"])),
                }
                break  # Baseline is same for all models

    return baseline_data


# ---------------------------------------------------------------------------
# Per-model summary data
# ---------------------------------------------------------------------------


def build_models_data(
    all_metrics: dict[str, list[dict]],
    model_keys: list["ModelKey"],
    baseline_data: dict[str, dict[str, float]],
    compute_baseline: bool,
) -> list[dict[str, Any]]:
    """Compute mean/std summary statistics per model for the bar-plot helpers.

    Collapses the flat metric lists from :func:`aggregate_seed_metrics` into
    ``{corr_mean, corr_std, ssim_mean, ssim_std, lpips_mean, lpips_std}``
    for each model/data-split pair.

    Parameters
    ----------
    all_metrics
        Aggregated per-model metric dictionaries keyed by example-set label.
        Each inner dict contains ``"correlations_100"``, ``"ssims_100"``,
        and ``"lpips_100"`` float lists.
    model_keys
        Ordered list of ``ModelKey``.  The order determines the order of
        entries in the returned list.
    baseline_data
        Pre-computed baseline mean/std statistics for ``"validation"`` and
        ``"rep2"`` splits.  Attached to each model entry for convenience.
    compute_baseline
        If ``False``, baseline entries in each model dict are set to ``None``.

    Returns
    -------
    models_data
        One dict per model with keys ``"model_key"``, ``"model_label"``,
        ``"validation"``, ``"rep2"``, ``"baseline_validation"``,
        ``"baseline_rep2"``.  The split dicts contain ``*_mean`` / ``*_std``
        floats for correlation, SSIM, and LPIPS.
    """
    _zero: dict[str, float] = {
        "corr_mean": 0.0,
        "corr_std": 0.0,
        "ssim_mean": 0.0,
        "ssim_std": 0.0,
        "lpips_mean": 0.0,
        "lpips_std": 0.0,
    }
    models_data: list[dict[str, Any]] = []

    for model_key in model_keys:
        model_entry: dict[str, Any] = {
            "model_key": model_key,
            "model_label": None,
            "validation": dict(_zero),
            "rep2": dict(_zero),
            "baseline_validation": baseline_data["validation"] if compute_baseline else None,
            "baseline_rep2": baseline_data["rep2"] if compute_baseline else None,
        }

        for split_key, metric_key in [
            ("validation", "validation_positions"),
            ("rep2", "rep_2_positions"),
        ]:
            for data in all_metrics[metric_key]:
                if data["model_key"] == model_key:
                    if model_entry["model_label"] is None:
                        model_entry["model_label"] = data["model_label"]
                    model_entry[split_key] = {
                        "corr_mean": float(np.mean(data["correlations_100"])),
                        "corr_std": float(np.std(data["correlations_100"])),
                        "ssim_mean": float(np.mean(data["ssims_100"])),
                        "ssim_std": float(np.std(data["ssims_100"])),
                        "lpips_mean": float(np.mean(data["lpips_100"])),
                        "lpips_std": float(np.std(data["lpips_100"])),
                    }
                    break

        models_data.append(model_entry)

    return models_data


# ---------------------------------------------------------------------------
# Comparison plots & summary
# ---------------------------------------------------------------------------


def create_comparison_plots_and_summary(
    models_data: list[dict[str, Any]],
    model_keys: list["ModelKey"],
    seeds_to_evaluate: list[int],
    baseline_data: dict[str, dict[str, float]],
    compute_baseline: bool,
) -> None:
    """Create comparison bar plots and log the summary table.

    Generates one bar plot per metric (correlation, SSIM, LPIPS) comparing
    all models on validation and rep-2 splits, and prints a formatted
    summary table.

    Parameters
    ----------
    models_data
        Per-model summary dicts each containing ``"validation"`` and
        ``"rep2"`` sub-dicts with ``*_mean`` / ``*_std`` floats.
    model_keys
        Ordered list of ``ModelKey``, one per model.  Used for axis labels
        and the legend text in each bar plot.  Must align positionally with
        ``models_data``.
    seeds_to_evaluate
        Seeds used during evaluation; displayed in titles when >1.
    baseline_data
        Baseline mean/std statistics for ``"validation"`` and ``"rep2"``.
        Shown as horizontal dashed lines when ``compute_baseline`` is True.
    compute_baseline
        Whether to overlay baseline reference lines on the bar plots.
    """
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.visualize.model_qc_plots import create_comparison_bar_plot
    from endo_pipeline.settings.workflow_defaults import (
        DEFAULT_MODEL_QC_LABELS,
        DEFAULT_MODEL_QC_MANIFEST_NAMES,
        DEFAULT_MODEL_QC_RUN_NAMES,
    )

    from .evaluation import ModelKey

    seed_suffix = f"_seeds_{len(seeds_to_evaluate)}" if len(seeds_to_evaluate) > 1 else ""
    comparison_output_path = get_output_path(
        "model_qc",
        "comparison",
        f"models_{len(model_keys)}{seed_suffix}",
    )

    # Determine model labels for bar-plot x-axis ticks.
    # The curated DEFAULT_MODEL_QC_LABELS list is a display override for the
    # specific 10-model latent-dimension sweep (pairs of manifest/run in
    # DEFAULT_MODEL_QC_MANIFEST_NAMES / DEFAULT_MODEL_QC_RUN_NAMES).  Only
    # use curated labels when *all* model keys are members of that sweep;
    # otherwise fall back to each model's own ``ModelKey.label`` (a two-line
    # ``manifest\nrun`` string) so ticks carry meaningful identifiers instead
    # of generic ``"Model N"``.
    _default_sweep_keys = {
        ModelKey(m, r)
        for m, r in zip(DEFAULT_MODEL_QC_MANIFEST_NAMES, DEFAULT_MODEL_QC_RUN_NAMES, strict=True)
    }
    if all(k in _default_sweep_keys for k in model_keys):
        # All models belong to the curated sweep; map each key to its
        # corresponding short label by positional order within the sweep.
        sweep_label_map = {
            ModelKey(m, r): lbl
            for m, r, lbl in zip(
                DEFAULT_MODEL_QC_MANIFEST_NAMES,
                DEFAULT_MODEL_QC_RUN_NAMES,
                DEFAULT_MODEL_QC_LABELS,
                strict=True,
            )
        }
        model_labels = [sweep_label_map[k] for k in model_keys]
    else:
        model_labels = [k.label for k in model_keys]

    seeds_info = (
        f" (averaged over {len(seeds_to_evaluate)} seeds)" if len(seeds_to_evaluate) > 1 else ""
    )

    metric_configs: list[tuple[str, str, str, dict[str, Any]]] = [
        ("corr", "Pearson Correlation (100% Noise)", "Correlation", {}),
        ("ssim", "SSIM Score (100% Noise)", "SSIM", {}),
        ("lpips", "LPIPS Score (100% Noise)", "LPIPS", {}),
    ]

    # Create comparison plots for each metric
    for metric_key, ylabel, title_base, extra_kw in metric_configs:
        create_comparison_bar_plot(
            models_data=models_data,
            metric_key=metric_key,
            ylabel=ylabel,
            title=f"{title_base}{seeds_info}",
            output_path=comparison_output_path,
            filename=f"{metric_key}_comparison_100_noise",
            model_labels=model_labels,
            show_baseline=compute_baseline,
            **extra_kw,
        )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY: Model Performance{seeds_info}")
    print("=" * 80)

    if compute_baseline and baseline_data["validation"]["corr_mean"] > 0:
        print("\nBASELINE (Temporal - Next Timepoint Comparison):")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            b = baseline_data[split_key]
            print(
                f"  {split_label} - Corr: {b['corr_mean']:.3f} ± {b['corr_std']:.3f}, "
                f"SSIM: {b['ssim_mean']:.3f} ± {b['ssim_std']:.3f}, "
                f"LPIPS: {b['lpips_mean']:.3f} ± {b['lpips_std']:.3f}"
            )
        print("-" * 80)

    for model_data in models_data:
        print(f"\n{model_data['model_label']}:")
        for split_label, split_key in [("Validation", "validation"), ("Rep2      ", "rep2")]:
            d = model_data[split_key]
            print(
                f"  {split_label} - Corr: {d['corr_mean']:.3f} ± {d['corr_std']:.3f}, "
                f"SSIM: {d['ssim_mean']:.3f} ± {d['ssim_std']:.3f}, "
                f"LPIPS: {d['lpips_mean']:.3f} ± {d['lpips_std']:.3f}"
            )
    print("=" * 80)
