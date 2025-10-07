def main(
    model_manifest_name: str = "diffae_finetuned_for_fixed",
    run_name: str | None = None,
    n_pcs: int = 3,
) -> None:
    """
    Validate integration of paired fixed/live data for integration of IF data.

    To do this, it does the following:
    1. Applies a fine-tuned diffAE model to extract features
    2. Projects the features into the reference PC space
    3. Constructs confidence ellipses to determine fixed/live PC mapping
        and uncertainty
    4. Plots the raw data for paired fixed and live PC values, confidence
        ellipses, linear model mapping between fixed and live data, and
        uncertainty.

    Parameters
    ----------
    model_manifest_name
        Name of model manifest to use for feature extraction for fixed data.
    run_name
        Name of model run to use for feature extraction for fixed data.
    n_pcs
        Number of PCs to validate.
    """

    from endo_pipeline import NUM_GPUS
    from endo_pipeline.configs import load_model_config
    from endo_pipeline.io import get_output_path
    from endo_pipeline.library.analyze.diffae_manifest import fit_pca
    from endo_pipeline.library.analyze.immunofluorescence import validate_pcs_for_integration
    from endo_pipeline.library.model import load_model_for_inference
    from endo_pipeline.library.visualize.integration import viz_validate_pcs_for_integration
    from endo_pipeline.manifests import load_model_manifest
    from endo_pipeline.settings import DIFFAE_MODEL_EVAL_FINETUNE_CONFIG

    # Results directory
    save_path = get_output_path("fixed_live_validation")

    reference_dataset_name: str = "20241217_20X"
    live_dataset_name: str = "20250214_pairedPreFixation"
    fixed_dataset_name: str = "20250214_pairedPostFixation"

    # Load diffAE model
    model_manifest = load_model_manifest(model_manifest_name)
    eval_config = load_model_config(DIFFAE_MODEL_EVAL_FINETUNE_CONFIG)
    model = load_model_for_inference(model_manifest, run_name, eval_config)

    # Set directory for aligned data
    model_save_path = get_output_path(
        "models", model_manifest_name, f"{fixed_dataset_name}_vs_{live_dataset_name}"
    )
    data_save_path = model_save_path / f"aligned_{fixed_dataset_name}_vs_{live_dataset_name}.csv"

    # Align paired fixed and live data and apply a diffAE model to extract features.
    fixed_features_path, live_features_path = (
        validate_pcs_for_integration.evaluate_model_paired_fixed_live(
            fixed_dataset_name,
            live_dataset_name,
            model_save_path,
            data_save_path,
            model,
            num_gpus=NUM_GPUS,
        )
    )

    # Load or fit reference PCA model and project features into reference PC space
    pca = fit_pca()

    # Project features from applying fine tuned diffAE model to fixed and live data into
    # reference PC space.
    fixed_features, live_features = (
        validate_pcs_for_integration.project_paired_fixed_live_data_into_ref_pc_space(
            pca, fixed_features_path, live_features_path
        )
    )

    lagged_ref_features, truncated_ref_features = (
        validate_pcs_for_integration.create_reference_timelapse_datasets(
            pca, reference_dataset_name
        )
    )

    for pc in range(1, n_pcs + 1):

        # Get common plot ranges for each PC
        axmin, axmax = viz_validate_pcs_for_integration.get_common_plot_range(
            fixed_features, live_features, lagged_ref_features, truncated_ref_features, pc
        )

        # Construct confidence ellipse to determine fixed/live PC mapping and uncertainty
        raw_data, validation_data = (
            validate_pcs_for_integration.get_paired_fixed_live_validation_features(
                pc, fixed_features, live_features
            )
        )

        # Construct confidence ellipse to determine live/ time-lagged live PC mapping
        # and uncertainty
        raw_data_ref, validation_data_ref = (
            validate_pcs_for_integration.get_paired_fixed_live_validation_features(
                pc, lagged_ref_features, truncated_ref_features
            )
        )

        # Plot raw data for paired fixed and live PC values as well as confidence ellipse,
        # linear model mapping between fixed and live data and uncertainty.
        viz_validate_pcs_for_integration.plot_paired_fixed_live_validation_features(
            save_path,
            pc,
            raw_data,
            validation_data,
            axmin=axmin,
            axmax=axmax,
        )

        # Plot raw data for paired live and time-lagged live PC values as well as confidence
        # ellipse, linear model mapping between live and time-lagged live data and uncertainty.
        viz_validate_pcs_for_integration.plot_paired_fixed_live_validation_features(
            save_path,
            pc,
            raw_data_ref,
            validation_data_ref,
            lagged_live_validation=True,
            axmin=axmin,
            axmax=axmax,
        )


if __name__ == "__main__":
    from endo_pipeline.__main__ import workflow_cli

    workflow_cli(main)
