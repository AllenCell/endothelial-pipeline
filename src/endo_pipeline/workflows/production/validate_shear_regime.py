tags = ["validation", "datasets"]


def main() -> None:
    """
    Validate shear regime annotations for dataset configs based on actual values present
    in the FlowConditions annotation.

    If any invalid shear regime annotations are found, update the shear_stress_regime field
    in each DatasetConfig and save the updated config.
    """
    import logging

    from endo_pipeline.configs import (
        get_regime_for_shear_stress,
        load_all_dataset_configs,
        save_dataset_config,
        validate_shear_stress_regime,
    )

    logger = logging.getLogger(__name__)

    for dataset_config in load_all_dataset_configs():
        flow_conditions = dataset_config.flow_conditions

        has_correct_regimes = []
        shear_stress_regimes = []
        for i, condition in enumerate(flow_conditions):
            # check if we have the correct annotation
            has_correct_shear_stress_regime = validate_shear_stress_regime(
                condition.shear_stress, dataset_config.shear_stress_regime[i]
            )
            has_correct_regimes.append(has_correct_shear_stress_regime)
            # if not, determine the correct annotation
            if not has_correct_shear_stress_regime:
                regime = get_regime_for_shear_stress(condition.shear_stress)
            else:
                regime = dataset_config.shear_stress_regime[i]
            shear_stress_regimes.append(regime)

        if all(has_correct_regimes):
            logger.info(
                "Shear stress regime annotation is correct for dataset [ %s ]",
                dataset_config.name,
            )
            continue
        else:
            dataset_config.shear_stress_regime = tuple(shear_stress_regimes)

            logger.info("Updated shear stress regime for dataset [ %s ]", dataset_config.name)
            save_dataset_config(dataset_config)
