from src.endo_pipeline.library.analyze.immunofluorescence import validate_pcs_for_integration
from src.endo_pipeline.library.visualize import viz_validate_pcs_for_integration

if __name__ == "__main__":
    """
    This workflow validates the integration of paired fixed and live data
    intended for integration of immunofluorescence data from fixed images
    into the PC space derived from live data.

    To do this, it does the following:
        1. Applies a fine-tuned diffAE model to extract features
        2. Projects the features into the reference PC space
        3. Constructs confidence ellipses to determine fixed/live PC mapping
           and uncertainty
        4. Plots the raw data for paired fixed and live PC values, confidence
              ellipses, linear model mapping between fixed and live data, and
              uncertainty.
    """

    live_dataset_name: str = "20250214_pairedPreFixation"
    fixed_dataset_name: str = "20250214_pairedPostFixation"
    model_name: str = "diffae_finetuned_for_fixed"
    n_pcs = 3

    # Align paired fixed and live data and apply a diffAE model to extract features.
    save_path, fixed_features_path, live_features_path = (
        validate_pcs_for_integration.apply_model_paired_fixed_live(
            fixed_dataset_name, live_dataset_name, model_name
        )
    )

    # Project features from applying fine tuned diffAE model to fixed and live data into
    # reference PC space.
    fixed_features, live_features = (
        validate_pcs_for_integration.project_paired_fixed_live_data_into_ref_PC_space(
            fixed_features_path, live_features_path
        )
    )

    # Create time-lagged live dataset to account for the time lag between fixed and live
    lagged_live_features, truncated_live_features = (
        validate_pcs_for_integration.create_time_lagged_live_dataset(live_features)
    )

    for pc in range(1, n_pcs + 1):

        # Construct confidence ellipse to determine fixed/live PC mapping and uncertainty
        raw_data, validation_data = (
            validate_pcs_for_integration.get_paired_fixed_live_validation_features(
                pc, fixed_features, live_features
            )
        )

        # Plot raw data for paired fixed and live PC values as well as confidence ellipse,
        # linear model mapping between fixed and live data and uncertainty.
        viz_validate_pcs_for_integration.plot_paired_fixed_live_validation_features(
            save_path,
            pc,
            raw_data,
            validation_data,
        )

        # Construct confidence ellipse to determine live/ time-lagged live PC mapping and uncertainty
        raw_data, validation_data = (
            validate_pcs_for_integration.get_paired_fixed_live_validation_features(
                pc, lagged_live_features, truncated_live_features
            )
        )

        # Plot raw data for paired live and time-lagged live PC values as well as confidence ellipse,
        # linear model mapping between live and time-lagged live data and uncertainty.
        viz_validate_pcs_for_integration.plot_paired_fixed_live_validation_features(
            save_path,
            pc,
            raw_data,
            validation_data,
            lagged_live_validation=True,
        )
