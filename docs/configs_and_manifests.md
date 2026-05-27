# Configs and manifests

This repository uses _configs_ to track information about inputs, such as dataset and dataset collections, and _manifests_ to track locations of outputs, such as images and dataframes.
This section provides details on the specific types of configs and manifests and lists patterns for loading these files.

---

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

---

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

## Dataset configs

Dataset configs are YAMLs located under `src/endo_pipeline/configs/datasets` that provide metadata about the dataset, such as identifying information (e.g. name, barcode) and experimental conditions and settings (e.g. microscope, objective, shear stress regime).

### Get list of dataset names

```python
from endo_pipeline.configs import get_available_dataset_names, get_datasets_in_collection

all_available_dataset_names = get_available_dataset_names()
dataset_names_in_collection = get_datasets_in_collection("collection_name")
```

### Load dataset config

```python
from endo_pipeline.configs import load_dataset_config

config = load_dataset_config("dataset_name")
```

## Dataframe manifests

Dataframe manifests are YAMLs located under `src/endo_pipeline/manifests/dataframes` that contain locations of dataframes and metadata about the workflow that produce the dataframes.
Dataframe manifests generally use dataset names as location keys, but other keys can be used (e.g. DiffAE training and validation dataframes are keyed as "training" and "validation").

Dataframes should only be loaded using the `load_dataframe` method, which takes an `DataframeLocation` object and allows us to assign multiple locations for a given image and flexibly select between them.

### Load dataframe from manifest

```python
from endo_pipeline.manifests import load_dataframe_manifest, get_dataframe_location_for_dataset
from endo_pipeline.io import load_dataframe

manifest = load_dataframe_manifest("manifest_name")
location = get_dataframe_location_for_dataset(manifest, "dataset_name")
dataframe = load_dataframe(location)
```

## Image manifests

Image manifests are YAMLs located under `src/endo_pipeline/manifests/images` that contains locations of images and metadata about the workflow that produce the images.
Image manifests generally use dataset names as location keys, but other keys can be used (e.g. live-fixed image registration uses a combined name).

Because images are often produced for each position or timepoint of a given dataset, rather than specifying the location of each image separately, locations in image manifest may contain a `{{position}}` and/or `{{timepoint}}` placeholder that can be dynamically set when loading an image for a specific position and/or timepoint.

Images should only be loaded using the `load_image` method, which takes an `ImageLocation` object and allows us to assign multiple locations for a given image and flexibly select between them.
The `load_image` method includes additional keyword options, including:

- `compute=True` (to return a NumPy array instead of a Dask array)
- `read=False` (to return a BioImage object without reading the image into memory)

### Load image from manifest

```python
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.manifests import load_image_manifest, get_image_location_for_dataset
from endo_pipeline.io import load_image

config = load_dataset_config("dataset_name")
manifest = load_image_manifest("manifest_name")
location = get_image_location_for_dataset(manifest, config, position=position, timepoint=timepoint)
image = load_image(location)
```

### Load zarr image from manifest

_When loading original Zarrs, which exist for all datasets that have a dataset config, you can use a dedicated utility method that will handle loading the `image_zarr` image manifest. Note that position is required when getting a Zarr location_

```python
from endo_pipeline.configs import load_dataset_config
from endo_pipeline.manifests import get_zarr_location_for_position
from endo_pipeline.io import load_image

config = load_dataset_config("dataset_name")
location = get_zarr_location_for_position(config, position=position)
image = load_image(location)
```

## Model manifests

Model manifests are YAMLs located under `src/endo_pipeline/manifests/models` that contain locations of trained models and metadata about the workflow that produced the models.

Models should only be loaded using the `load_model` method, which takes a `ModelLocation` object and allows us to assign multiple locations for a given model and flexibly select between them.
Note that the `load_model` currently only works for models tracked by MLflow.
The `load_model` method includes additional keyword options, including:

- `instantiate=True` (to return an instantiated model object instead of a CytoDLModel)

### Load DiffAE model from manifest

```python
from endo_pipeline.manifests import load_model_manifest, get_model_location_for_run
from endo_pipeline.io import load_model

manifest = load_model_manifest("model_name")
location = get_model_location_for_run(manifest, "run_name")
model = load_model(location)
```

### Load CellPose model from manifest

_Loading the CellPose model is currently handled directly and is not integrated into the `load_model` functionality. We will likely update this behavior to match how DiffAE models are loaded._

```python
from endo_pipeline.io import load_model
from endo_pipeline.manifests import load_model_manifest, get_model_location_for_run
from endo_pipeline.settings.workflow_defaults import LABELFREE_NUCLEI_MODEL_MANIFEST_NAME, LABELFREE_NUCLEI_MODEL_RUN_NAME

manifest = load_model_manifest(LABELFREE_NUCLEI_MODEL_MANIFEST_NAME)
location = get_model_location_for_run(manifest, LABELFREE_NUCLEI_MODEL_RUN_NAME)
model = load_model(location)
```
