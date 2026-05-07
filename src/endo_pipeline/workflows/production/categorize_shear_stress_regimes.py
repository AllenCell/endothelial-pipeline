def main() -> None:
    """
    Categorizes shear stress regime based on actual values in flow conditions.

    #validation #datasets

    If any invalid shear regime annotations are found, the dataset config will
    be re-saved with the correct regime.
    """

    import logging

    from endo_pipeline.configs import (
        get_regime_for_shear_stress,
        load_all_dataset_configs,
        save_dataset_config,
    )

    logger = logging.getLogger(__name__)

    for dataset_config in load_all_dataset_configs():
        shear_stress_regimes = []

        for condition in dataset_config.flow_conditions:
            regime = get_regime_for_shear_stress(condition.shear_stress)
            shear_stress_regimes.append(regime)

        if dataset_config.shear_stress_regime == shear_stress_regimes:
            logger.info("Shear stress regime for dataset [ %s ] is correct", dataset_config.name)
        else:
            print(shear_stress_regimes, dataset_config.shear_stress_regime)
            logger.warning("Updated shear stress regime for dataset [ %s ]", dataset_config.name)
            dataset_config.shear_stress_regime = shear_stress_regimes
            save_dataset_config(dataset_config)
