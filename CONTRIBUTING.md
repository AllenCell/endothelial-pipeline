# Contributing guidelines

### Pre-commit plugin

Use [pre-commit](https://pre-commit.com/) to automatically format your code to match the recommended coding style.
You will only need to do the following steps once.

**1. Install the development dependencies**

```bash
uv sync --only-dev --inexact
```

This command will install the development dependencies without affecting installations of optional dependency groups.

**2. Install the git hooks**

```bash
pre-commit install
```

The plugin will now run every time you commit changes.
If any of the hooks fail, or if the hooks alters files that are part of the commit, the commit will be rejected until you address the error(s) and/or stage the change(s).

If necessary, you may bypass the hooks for the commit:

```bash
git commit --no-verify -m "commit message"
```

### Multiple virtual environments

If you want to run workflows on different systems with code stored at the same location, we recommend defining dedicated virtual environments for each system.
This will ensure that `uv` correctly installs packages and, if needed, links them to system libraries.

By default `uv` will create the virtual environment for the project at `.venv`.
Instead, for each different system, specify the `UV_PROJECT_ENVIRONMENT` environment variable before syncing virtual environment.
For example:

**Slurm**

```bash
export UV_PROJECT_ENVIRONMENT=.venv_slurm
uv sync
```

**A100s**

```bash
export UV_PROJECT_ENVIRONMENT=.venv_a100s
uv sync
```

This creates two separate environments, one for a Slurm cluster and one for A100s machines.
You can then activate the appropriate environment, or use `uv run`, which will use the environment specified by `UV_PROJECT_ENVIRONMENT`.
You can check which environment `uv` is using with:

```bash
echo $UV_PROJECT_ENVIRONMENT
```

If no value is returned, `uv` will default to using `.venv`.

If you often switch between systems, it may be helpful to add the `export` statement in your `.bashrc` (or equivalent) shell configuration file to automatically set the environment variable.

### Configs and manifests

This repository uses _configs_ to track information about inputs, such as dataset and dataset collections, and _manifests_ to track locations of outputs, such as images and dataframes.
This section provides details on the specific types of configs and manifests and lists patterns for loading these files.

- [Dataset configs](#dataset-configs)
    - [Get list of dataset names](#get-list-of-dataset-names)
    - [Load dataset config](#load-dataset-config)
- [Dataframe manifests](#dataframe-manifests)
    - [Load dataframe from manifest](#load-dataframe-from-manifest)
- [Image manifests](#image-manifests)
    - [Load image from manifest](#load-image-from-manifest)
    - [Load zarr image from manifest](#load-zarr-image-from-manifest)
- [Model manifests](#model-manifests)
    - [Load DiffAE model from manifest](#load-diffae-model-from-manifest)
    - [Load CellPose model from manifest](#load-cellpose-model-from-manifest)

You will often need to iterate through datasets and load both configs and manifests.
We recommend the following pattern (using an image manifest as an example) to avoid redundant loading.
The manifest is loaded once outside the loop, and the config and locations are loaded inside the loop.

```python
from endo_pipeline.configs import load_dataset_config, get_datasets_in_collection
from endo_pipeline.manifests import load_image_manifest, get_image_location_for_dataset
from endo_pipeline.io import load_image

image_manifest = load_image_manifest("manifest_name")
list_of_dataset_names = get_datasets_in_collection("collection_name")

for dataset_name in list_of_dataset_names:
    dataset_config = load_dataset_config(dataset_name)
    image_location = get_image_location_for_dataset(image_manifest, dataset_config)
    image = load_image(image_location)
```

#### Dataset configs

Dataset configs are YAMLs located under `src/endo_pipeline/configs/datasets` that provide metadata about the dataset, such as identifying information (e.g. name, barcode) and experimental conditions and settings (e.g. microscope, objective, shear stress regime).

##### Get list of dataset names

```python
from endo_pipeline.configs import get_available_dataset_names, get_datasets_in_collection

all_available_dataset_names = get_available_dataset_names()
dataset_names_in_collection = get_datasets_in_collection("collection_name")
```

##### Load dataset config

```python
from endo_pipeline.configs import load_dataset_config

config = load_dataset_config("dataset_name")
```

#### Dataframe manifests

Dataframe manifests are YAMLs located under `src/endo_pipeline/manifests/dataframes` that contain locations of dataframes and metadata about the workflow that produce the dataframes.
Dataframe manifests generally use dataset names as location keys, but other keys can be used (e.g. DiffAE training and validation dataframes are keyed as "training" and "validation").

Dataframes should only be loaded using the `load_dataframe` method, which takes an `DataframeLocation` object and allows us to assign multiple locations for a given image and flexibly select between them.

##### Load dataframe from manifest

```python
from endo_pipeline.manifests import load_dataframe_manifest, get_dataframe_location_for_dataset
from endo_pipeline.io import load_dataframe

manifest = load_dataframe_manifest("manifest_name")
location = get_dataframe_location_for_dataset(manifest, "dataset_name")
dataframe = load_dataframe(location)
```

#### Image manifests

Image manifests are YAMLs located under `src/endo_pipeline/manifests/images` that contains locations of images and metadata about the workflow that produce the images.
Image manifests generally use dataset names as location keys, but other keys can be used (e.g. live-fixed image registration uses a combined name).

Because images are often produced for each position or timepoint of a given dataset, rather than specifying the location of each image separately, locations in image manifest may contain a `{{position}}` and/or `{{timepoint}}` placeholder that can be dynamically set when loading an image for a specific position and/or timepoint.

Images should only be loaded using the `load_image` method, which takes an `ImageLocation` object and allows us to assign multiple locations for a given image and flexibly select between them.
The `load_image` method includes additional keyword options, including:

- `compute=True` (to return a NumPy array instead of a Dask array)
- `read=False` (to return a BioImage object without reading the image into memory)

##### Load image from manifest

```python
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.manifests import load_image_manifest, get_image_location_for_dataset
from endo_pipeline.io import load_image

config = load_dataset_config("dataset_name")
manifest = load_image_manifest("manifest_name")
location = get_image_location_for_dataset(manifest, config, position=position, timepoint=timepoint)
image = load_image(location)
```

##### Load zarr image from manifest

_When loading original Zarrs, which exist for all datasets that have a dataset config, you can use a dedicated utility method that will handle loading the `image_zarr` image manifest. Note that position is required when getting a Zarr location_

```python
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.io import load_image

config = load_dataset_config("dataset_name")
location = get_zarr_location_for_position(config, position=position)
image = load_image(location)
```

#### Model manifests

Model manifests are YAMLs located under `src/endo_pipeline/manifests/models` that contain locations of trained models and metadata about the workflow that produced the models.

Models should only be loaded using the `load_model` method, which takes a `ModelLocation` object and allows us to assign multiple locations for a given model and flexibly select between them.
Note that the `load_model` currently only works for models tracked by MLflow.
The `load_model` method includes additional keyword options, including:

- `instantiate=True` (to return an instantiated model object instead of a CytoDLModel)

##### Load DiffAE model from manifest

```python
from endo_pipeline.manifests import load_model_manifest, get_model_location_for_run
from endo_pipeline.io import load_model

manifest = load_model_manifest("model_name")
location = get_model_location_for_run(manifest, "run_name")
model = load_model(location)
```

##### Load CellPose model from manifest

_Loading the CellPose model is currently handled directly and is not integrated into the `load_model` functionality. We will likely update this behavior to match how DiffAE models are loaded._

```python
from cellpose.models import CellposeModel
from endo_pipeline.manifests import load_model_manifest, get_model_location_for_run

manifest = load_model_manifest("nuc_pred_labelfree")
location = get_model_location_for_run(manifest, "finetuned_20250419")
model = CellposeModel(pretrained_model=location.path.as_posix())
```

## Model training and evaluation

These guidelines aim to support flexible and reproducible use of the model training and evaluation workflows.

### Model identifiers

There are two main identifiers for models. These identifiers may be passed to the appropriate workflows via the CLI to select models. Otherwise, the default names will be used.

| Identifier | Description | CLI | Default |
| - | - | - | - |
| **model manifest name** | general name for a set of models | `--model-manifest-name MODEL_MANIFEST_NAME` | `diffae_baseline_exclude_cell_piling` |
| **run name** | specific name of a model training run | `--run-name RUN_NAME` | `20251110_latent_512` |

_Note that **model manifest name** is analogous to MLflow "experiments" and **run name** is analogous to MLflow "runs" (these mappings are reflected in the model configs)._

### Model training options

There are currently three main model training options. These options may be passed to the appropriate workflows via the CLI. Where indicated, these options will also be used to format default manifest names.

| Option | Description | CLI | Formatting |
| - | - | - | - |
| **image crop size** | length of the 2D image crop in pixels used for training | `--crop-size CROP_SIZE` | `patch_CROP_SIZExCROP_SIZE` |
| **cell piling timepoints** | if timepoints annotated as cell piling are included in the training | `--include-cell-piling` or `--exclude-cell-piling` | `include_cell_piling` or `exclude_cell_piling` |
| **conditioning image type** | name of image type to use for semantic conditioning (`cdh5` or `bf`) | `--condition-on CONDITION_IMG_NAME` | `condition_on_CONDITION_IMG_NAME`

### Model evaluation options

There is currently one main model evaluation option. This options may be passed to the appropriate workflows via the CLI. Where indicated, these options will also be used to format default manifest names.

| Option | Description | CLI | Options |
| - | - | - | - |
| **crop pattern** | select grid-based or track-based crops | `--crop-pattern CROP_PATTERN` | `grid` or `tracked` |


### Model training workflows

<sup>
:purple_circle: = item or change that must be merged into the repo via a PR<br />
:white_circle: = item or change that should not be merged into the repo
</sup>

#### 1. Build the training and validation dataframes

- workflow = `create-diffae-training-dataframe`
- input (training options) = **cell piling timepoints**
- :purple_circle: output = training dataframe manifest at `src/endo_pipeline/manifests/dataframes/diffae_training_dataframe_PILING`

#### 2. Build the model training config

- workflow = `build-diffae-train-config`
- input (identifier options) = **model manifest name**, **run_name**
- input (training options) = **cell piling timepoints**, **image crop size**, **conditioning image type**, **latent space dimension**
- :white_circle: output = resolved model config at `results/models/MODEL_MANIFEST_NAME/RUN_NAME/configs/train.yaml`
- :purple_circle: output = updated model manifest with pending training run at `src/endo_pipeline/manifests/models/diffae_PATCH_CONDITIONING_LATENT_PILING` (recommendation is to open a draft PR with this change until the next step is complete)

#### 3. Train the model

- workflow = `train-diffae`
- input (identifier options) = **model manifest name**, **run_name**
- :white_circle: output = model training run on MLflow
- :purple_circle: output = updated model manifest with MLflow run id at `src/endo_pipeline/manifests/models/diffae_PATCH_CONDITIONING_LATENT_PILING`

### Model evaluation workflows

<sup>
:purple_circle: = item or change that must be merged into the repo via a PR<br />
:white_circle: = item or change that should not be merged into the repo
</sup>

#### 1. Build the evaluation dataframe

- workflow = `create-diffae-eval-dataframe`
- input (evaluation options) = **crop pattern**
- :purple_circle: output = evaluation dataframe manifest at `src/endo_pipeline/manifests/dataframes/diffae_evaluation_dataframe_CROP_PATTERN`

#### 2. Build the model evaluation config

- workflow = `build-diffae-eval-config`
- input (identifier options) = **model manifest name**, **run name**
- input (evaluation options) = **crop pattern**
- :white_circle: output = resolved model config for each dataset at `results/models/MODEL_MANIFEST_NAME/RUN_NAME/configs/eval_CROP_DATASET.yaml`
- :purple_circle: output = updated model manifest with pending evaluation run(s) at `src/endo_pipeline/manifests/models/diffae_(MODEL_MANIFEST_NAME)_(RUN_NAME)_(CROP_PATTERN)` (recommendation is to open a draft PR with this change until the next step is complete)

#### 3. Evaluate the model

- workflow = `eval-diffae`
- input (identifier options) = **model manifest name**, **run_name**
- input (evaluation options) = **crop pattern**
- :purple_circle: output = updated model manifest with MLflow run id at `src/endo_pipeline/manifests/models/diffae_(MODEL_MANIFEST_NAME)_(RUN_NAME)_(CROP_PATTERN)`
