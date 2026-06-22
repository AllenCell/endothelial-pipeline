# Model training and evaluation

These guidelines aim to support flexible and reproducible use of the model training and evaluation workflows.

---

- [Model identifiers](#model-identifiers)
- [Model training options](#model-training-options)
- [Model evaluation options](#model-evaluation-options)
- [Model training workflows](#model-training-workflows)
    - [Build the training and validation dataframes](#1-build-the-training-and-validation-dataframes)
    - [Build the model training config](#2-build-the-model-training-config)
    - [Train the model](#3-train-the-model)
- [Model evaluation workflows](#model-evaluation-workflows)
    - [Build the evaluation dataframe](#1-build-the-evaluation-dataframe)
    - [Build the model evaluation config](#2-build-the-model-evaluation-config)
    - [Evaluate the model](#3-evaluate-the-model)

---

## Model identifiers

There are two main identifiers for models. These identifiers may be passed to the appropriate workflows via the CLI to select models. Otherwise, the default names will be used.

| Identifier | Description | CLI | Default |
| - | - | - | - |
| **model manifest name** | general name for a set of models | `--model-manifest-name MODEL_MANIFEST_NAME` | `diffae_baseline` |
| **run name** | specific name of a model training run | `--run-name RUN_NAME` | `latent_512` |

_Note that **model manifest name** is analogous to MLflow "experiments" and **run name** is analogous to MLflow "runs" (these mappings are reflected in the model configs)._

## Model training options

There are currently three main model training options. These options may be passed to the appropriate workflows via the CLI. Where indicated, these options will also be used to format default manifest names.

| Option | Description | CLI | Formatting |
| - | - | - | - |
| **image crop size** | length of the 2D image crop in pixels used for training | `--crop-size CROP_SIZE` | `patch_CROP_SIZExCROP_SIZE` |
| **conditioning image type** | name of image type to use for semantic conditioning (`cdh5` or `bf`) | `--condition-on CONDITION_IMG_NAME` | `condition_on_CONDITION_IMG_NAME` |
| **latent space dimension** | number of latent dimensions | `--latent-dim LATENT_DIM_SIZE` | `latent_LATENT_DIM_SIZE` |

## Model evaluation options

There is currently one main model evaluation option. This options may be passed to the appropriate workflows via the CLI. Where indicated, these options will also be used to format default manifest names.

| Option | Description | CLI | Options |
| - | - | - | - |
| **crop pattern** | select grid-based or track-based crops | `--crop-pattern CROP_PATTERN` | `grid` or `tracked` |

## Model training workflows

<sup>
:purple_circle: = item or change that must be merged into the repo via a PR<br />
:white_circle: = item or change that should not be merged into the repo
</sup>

### 1. Build the training and validation dataframes

- workflow = `create-diffae-train-dataframe`
- input (training options) = (none)
- :purple_circle: output = training dataframe manifest at `src/endo_pipeline/manifests/dataframes/diffae_training_dataframe`

### 2. Build the model training config

- workflow = `build-diffae-train-config`
- input (identifier options) = **model manifest name**, **run_name**
- input (training options) = **image crop size**, **conditioning image type**, **latent space dimension**
- :white_circle: output = resolved model config at `results/models/MODEL_MANIFEST_NAME/RUN_NAME/configs/train.yaml`
- :purple_circle: output = updated model manifest with pending training run at `src/endo_pipeline/manifests/models/diffae_PATCH_CONDITIONING_LATENT` (recommendation is to open a draft PR with this change until the next step is complete)

### 3. Train the model

- workflow = `train-diffae`
- input (identifier options) = **model manifest name**, **run_name**
- :white_circle: output = model training run on MLflow
- :purple_circle: output = updated model manifest with MLflow run id at `src/endo_pipeline/manifests/models/diffae_PATCH_CONDITIONING_LATENT`

## Model evaluation workflows

<sup>
:purple_circle: = item or change that must be merged into the repo via a PR<br />
:white_circle: = item or change that should not be merged into the repo
</sup>

### 1. Build the evaluation dataframe

- workflow = `create-diffae-eval-dataframe`
- input (evaluation options) = **crop pattern**
- :purple_circle: output = evaluation dataframe manifest at `src/endo_pipeline/manifests/dataframes/diffae_evaluation_dataframe_CROP_PATTERN`

### 2. Build the model evaluation config

- workflow = `build-diffae-eval-config`
- input (identifier options) = **model manifest name**, **run name**
- input (evaluation options) = **crop pattern**
- :white_circle: output = resolved model config for each dataset at `results/models/MODEL_MANIFEST_NAME/RUN_NAME/configs/eval_CROP_DATASET.yaml`
- :purple_circle: output = updated model manifest with pending evaluation run(s) at `src/endo_pipeline/manifests/models/diffae_(MODEL_MANIFEST_NAME)_(RUN_NAME)_(CROP_PATTERN)` (recommendation is to open a draft PR with this change until the next step is complete)

### 3. Evaluate the model

- workflow = `eval-diffae`
- input (identifier options) = **model manifest name**, **run_name**
- input (evaluation options) = **crop pattern**
- :purple_circle: output = updated model manifest with MLflow run id at `src/endo_pipeline/manifests/models/diffae_(MODEL_MANIFEST_NAME)_(RUN_NAME)_(CROP_PATTERN)`
