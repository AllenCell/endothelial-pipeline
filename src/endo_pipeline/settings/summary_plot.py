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
        "20260309_20X",  # CDH5 CD31-sorted under low shear stress
        "20250908_20X",  # CDH5 Ex3 deletion under low shear stress, not migratory
        "20251029_20X",  # CDH5 Ex3 deletion under low shear stress
        "20251119_20X",  # CDH5 Ex3 deletion under low shear stress
        "20260325_20X",  # CDH5 Ex3 deletion under low shear stress
    ],
}

CELL_LINE_LABEL_MAP = {
    "AICS-126 cl. 41": "Parental",
    "AICS-126 cl. 41 CD31-sorted": "Control",
    "AICS-177 cl. 26": "Ex3Del",
}

COLOR_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#882255",  # wine
    "#AA4499",  # magenta
    "#117733",  # forest green
    "#CC6677",  # rose
    "#6699CC",  # steel blue
    "#661100",  # dark brown
    "#999933",  # olive
    "#AA4466",  # raspberry
    "#44BB99",  # mint
    "#BBCC33",  # pear
    "#EE8866",  # peach
]

# Collect all unique datasets across all groups (preserving first-appearance order)
_ALL_DATASETS: list[str] = []
for _ds_list in SUMMARY_PLOT_DATASETS.values():
    for _ds in _ds_list:
        if _ds not in _ALL_DATASETS:
            _ALL_DATASETS.append(_ds)

DATASET_COLOR_MAP: dict[str, str] = {
    ds: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, ds in enumerate(_ALL_DATASETS)
}
