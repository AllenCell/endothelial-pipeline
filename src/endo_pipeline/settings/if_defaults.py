PLOT_FEAT_COLS: list[str] = [
    # "SMAD1_sum_sum_proj",
    "SMAD1_norm_area_sum_sum_proj",
]  # "SMAD1_mean_sum_proj"
"""Smad1 feature columns for plotting."""

PLOT_FEAT_NAMES: list[str] = [
    # "Total SMAD1 intensity in nuclear mask volume",
    "Total SMAD1 intensity / N pixels \nin nuclear mask volume",
]  # "SMAD1 mean intensity of sum projection\nin nuclear mask"
"""Smad1 feature names for plotting."""

DATASET_GROUPS: dict[str, list[tuple[list[str], str]]] = {
    "20250509": [
        (
            [
                "20250509_20X_IF3",
                "20250509_20X_IF5",
                "20250509_20X_IF7",
                "20250509_20X_IF9",
            ],
            "24hr_low_density_varied_shear_stress",
        ),
        (
            [
                "20250509_20X_IF2",
                "20250509_20X_IF12",
                "20250509_20X_IF1",
            ],
            "24hr_high_density_varied_shear_stress",
        ),
    ],
    "20250522": [
        (["20250522_20X_IFH", "20250522_20X_IFJ"], "24hr_low_density_varied_shear_stress"),
        (
            ["20250522_20X_IFI", "20250522_20X_IFG", "20250522_20X_IFN", "20250522_20X_IFM"],
            "48hr_varied_shear_stress",
        ),
        (
            [
                "20250522_20X_IFH",
                "20250522_20X_IFA",
                "20250522_20X_IFB",
                "20250522_20X_IFD",
                "20250522_20X_IFC",
                "20250522_20X_IFF",
                "20250522_20X_IFE",
                "20250522_20X_IFG",
            ],
            "24hr_low_plus_X_high_over_time",
        ),
        (
            [
                "20250522_20X_IFJ",
                "20250522_20X_IFK",
                "20250522_20X_IFL",
                "20250522_20X_IFM",
                "20250522_20X_IFN",
            ],
            "24hr_high_plus_X_low_over_time",
        ),
    ],
    "20250929": [
        (["20250929_20X_IF0", "20250929_20X_IF2", "20250929_20X_IF9"], "no_shear_stress_over_time"),
        (["20250929_20X_IF2", "20250929_20X_IF1", "20250929_20X_IF3"], "24hr_varied_shear_stress"),
        (["20250929_20X_IF9", "20250929_20X_IF8", "20250929_20X_IF10"], "48hr_varied_shear_stress"),
        (
            [
                "20250929_20X_IF1",
                "20250929_20X_IF4",
                "20250929_20X_IF5",
                "20250929_20X_IF6",
                "20250929_20X_IF8",
            ],
            "24hr_low_plus_X_high_over_time",
        ),
        (
            [
                "20250929_20X_IF3",
                "20250929_20X_IF7",
                "20250929_20X_IF10",
            ],
            "int_shear_stress_over_time",
        ),
    ],
    "20251103": [
        (["20251103_20X_IF0", "20251103_20X_IF2"], "no_shear_stress_over_time"),
        (["20251103_20X_IF2", "20251103_20X_IF3", "20251103_20X_IF1"], "24hr_varied_shear_stress"),
        (["20251103_20X_IF8", "20251103_20X_IF10"], "48hr_varied_shear_stress"),
        (
            [
                "20251103_20X_IF1",
                "20251103_20X_IF4",
                "20251103_20X_IF5",
                "20251103_20X_IF6",
                "20251103_20X_IF8",
            ],
            "24hr_high_plus_X_low_over_time",
        ),
        (
            [
                "20251103_20X_IF3",
                "20251103_20X_IF7",
                "20251103_20X_IF10",
            ],
            "medium_shear_stress_over_time",
        ),
    ],
}
"""Dataset groupings for SMAD1 analysis by experiment date."""
