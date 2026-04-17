"""Summary plot settings"""

SUMMARY_PLOT_DATASETS: dict[str, list[str]] = {
    "low_high": [
        "20250618_20X",  # 6 dyn / cm2
        "20250402_20X",  # 6 dyn / cm2
        "20250409_20X",  # 6 dyn / cm2
        "20250611_20X",  # 20 dyn / cm2
        "20251001_20X",  # 20 dyn / cm2
    ],
    "intermediate": [
        "20250618_20X",  # 6 dyn / cm2
        "20250402_20X",  # 6 dyn / cm2
        "20250409_20X",  # 6 dyn / cm2
        "20250428_20X",  # 9 dyn / cm2
        "20250716_20X",  # 9 dyn / cm2
        "20250319_20X",  # 12 dyn / cm2
        "20250604_20X",  # 12 dyn / cm2
        "20260121_20X",  # 12 dyn / cm2
        "20260126_20X",  # 12 dyn / cm2
        "20260209_20X",  # 12 dyn / cm2
        "20260216_20X",  # 12 dyn / cm2
        "20250326_20X",  # 15 dyn / cm2
        "20250813_20X",  # 15 dyn / cm2
        "20260114_20X",  # 15 dyn / cm2
        "20260128_20X",  # 15 dyn / cm2
        "20260202_20X",  # 15 dyn / cm2
        "20260204_20X",  # 15 dyn / cm2
        "20260211_20X",  # 16 dyn / cm2
        "20260218_20X",  # 16 dyn / cm2
        "20260225_20X",  # 16 dyn / cm2
        "20260302_20X",  # 16 dyn / cm2
        "20250611_20X",  # 20 dyn / cm2
        "20251001_20X",  # 20 dyn / cm2
    ],
    "perturbation": [
        "20250618_20X",  # 6 dyn / cm2
        "20250402_20X",  # 6 dyn / cm2
        "20250409_20X",  # 6 dyn / cm2
        "20251105_20X",  # isogenic control CD31 sorted under low shear stress
        "20260309_20X",  # isogenic control CD144 sorted under low shear stress
        "20251029_20X",  # CDH5 Ex2 deletion under low shear stress
        "20251119_20X",  # CDH5 Ex2 deletion under low shear stress
        "20260325_20X",  # CDH5 Ex2 deletion under low shear stress
    ],
    "perturbation_supp": [
        "20250618_20X",  # 6 dyn / cm2
        "20250402_20X",  # 6 dyn / cm2
        "20250409_20X",  # 6 dyn / cm2
        "20251022_20X",  # isogenic control CD31 sorted under low shear stress, exclude from main figure
        "20251105_20X",  # isogenic control CD31 sorted under low shear stress
        "20260309_20X",  # isogenic control CD144 sorted under low shear stress
        "20250908_20X",  # CDH5 Ex2 deletion under low shear stress, exclude from main figure
        "20251029_20X",  # CDH5 Ex2 deletion under low shear stress
        "20251119_20X",  # CDH5 Ex2 deletion under low shear stress
        "20260325_20X",  # CDH5 Ex2 deletion under low shear stress
    ],
}
