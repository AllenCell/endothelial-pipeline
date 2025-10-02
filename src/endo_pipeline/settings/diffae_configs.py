"""Global settings for template DiffAE model configurations."""

RELATIVE_PATH_TO_MODEL_DIR = "src/endo_pipeline/library/model/"
"""Path to the model directory relative to the root of the repository."""

EVAL_CONFIG = "diffae_eval.yaml"
"""Name of the template configuration file for evaluating a DiffAE model."""

TRAIN_CONFIG = "diffae_train.yaml"
"""Name of the template configuration file for training a DiffAE model."""

FINETUNE_CONFIG = "diffae_finetune.yaml"
"""Name of the template configuration file for finetuning a DiffAE model."""

LEGACY_CONFIG = "diffae_04_10_eval.yaml"
"""Name of the configuration file for evaluating the legacy DiffAE model diffae_04_10."""

RELATIVE_PATH_TO_EVAL_CONFIG = f"{RELATIVE_PATH_TO_MODEL_DIR}{EVAL_CONFIG}"
"""Relative path to the template configuration file for evaluating a DiffAE model."""

RELATIVE_PATH_TO_TRAIN_CONFIG = f"{RELATIVE_PATH_TO_MODEL_DIR}{TRAIN_CONFIG}"
"""Relative path to the template configuration file for training a DiffAE model."""

RELATIVE_PATH_TO_FINETUNE_CONFIG = f"{RELATIVE_PATH_TO_MODEL_DIR}{FINETUNE_CONFIG}"
"""Relative path to the template configuration file for finetuning a DiffAE model."""

RELATIVE_PATH_TO_LEGACY_CONFIG = f"{RELATIVE_PATH_TO_MODEL_DIR}{LEGACY_CONFIG}"
"""Relative path to the configuration file for evaluating the legacy DiffAE model diffae_04_10."""
