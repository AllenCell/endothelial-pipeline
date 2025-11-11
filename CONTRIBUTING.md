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

## Model training and evaluation

These guidelines aim to support flexible and reproducible use of the model training and evaluation workflows.

### Model identifiers

There are two main identifiers for models. These identifiers may be passed to the appropriate workflows via the CLI to select models. Otherwise, the default names will be used.

| Identifier | Description | CLI | Default |
| - | - | - | - |
| **model manifest name** | general name for a set of models | `--model-manifest-name MODEL_MANIFEST_NAME` | `diffae_RESOLUTION_PATCH_PILING` |
| **run name** | specific name of a model training run | `--run-name RUN_NAME` | `diffae_YYYYMMDD_MMHHSS` |

_Note that **model manifest name** is analogous to MLflow "experiments" and **run name** is analogous to MLflow "runs" (these mappings are reflected in the model configs)._

### Model training options

There are currently three main model training options. These options may be passed to the appropriate workflows via the CLI. Where indicated, these options will also be used to format default manifest names.

| Option | Description | CLI | Formatting |
| - | - | - | - |
| **zarr resolution level** | resolution level of the Zarr files used for training | `--resolution-level RESOLUTION_LEVEL` | `resolution_<RESOLUTION_LEVEL>` |
| **image crop size** | length of the 2D image crop in pixels used for training | `--crop-size CROP_SIZE` | `patch_<CROP_SIZE>x<CROP_SIZE>` |
| **cell piling timepoints** | if timepoints annotated as cell piling are included in the training | `--include-cell-piling` or `--exclude-cell-piling` | `include_cell_piling` or `exclude_cell_piling` |

### Model training workflows

<sup>
:purple_circle: = item or change that must be merged into the repo via a PR<br />
:white_circle: = item or change that should not be merged into the repo
</sup>

#### 1. Build the training and validation dataframes

- workflow = `create-diffae-training-dataframe`
- input (training options) = **zarr resolution level**, **cell piling timepoints**
- :purple_circle: output = training dataframe manifest at `src/endo_pipeline/manifests/dataframes/diffae_training_dataframe_RESOLUTION_PILING`

#### 2. Build the model training config

- workflow = `build-diffae-train-config`
- input (identifier options) = **model manifest name**, **run_name**
- input (training options) = **zarr resolution level**, **cell piling timepoints**, **image crop size**, **conditioning image type**
- :white_circle: output = resolved model config at `results/models/MODEL_MANIFEST_NAME/RUN_NAME/configs/train.yaml`
- :purple_circle: output = updated model manifest with pending training run at `src/endo_pipeline/manifests/models/diffae_RESOLUTION_PATCH_CONDITIONING_PILING` (recommendation is to open a draft PR with this change until the next step is complete)

#### 3. Train the model

- workflow = `train-diffae`
- input (identifier options) = **model manifest name**, **run_name**
- :white_circle: output = model training run on MLflow
- :purple_circle: output = updated model manifest with MLflow run id at `src/endo_pipeline/manifests/models/diffae_RESOLUTION_PATCH_CONDITIONING_PILING`
