tags = ["datasets"]


def main() -> None:
    """
    Update shear regime annotations for dataset configs based on actual
    values present in the FlowConditions annotation.
    """

    from endo_pipeline.configs import (
        get_regime_for_shear_stress,
        load_all_dataset_configs,
        save_dataset_config,
    )

    for dataset_config in load_all_dataset_configs():
        flow_conditions = dataset_config.flow_conditions
        shear_stress_regimes = []

        for condition in flow_conditions:
            regime = get_regime_for_shear_stress(condition.shear_stress)
            shear_stress_regimes.append(regime)

        if len(flow_conditions) == 1:
            dataset_config.shear_stress_regime = next(iter(shear_stress_regimes))
        else:
            dataset_config.shear_stress_regime = tuple(shear_stress_regimes)

        save_dataset_config(dataset_config)
