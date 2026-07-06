# Dynamics of ML-based morphological features indicate a shear stress-dependent bifurcation of hiPSC-derived endothelial cell states

The code in this repository runs all analyses and generates all figures, tables, and movies for [XXX et al.](DOI).
It is primarily intended to support reproducibility of our research.
In addition, researchers may find parts of this code valuable for future work.

> [!NOTE]
> The `main` branch reflects the most up-to-date version of the code.
> To exactly reproduce the contents from the [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.06.28.601071v1) version of the manuscript, use the [`bioRxiv-v1` release](https://github.com/AllenCell/endothelial-pipeline/releases/tag/bioRxiv-v)

We release all timelapse data used in this study in the OME-Zarr format to democratize their access.
The data are publicly available on [S3](https://open.quiltdata.com/b/allencell/tree/aics/endo_cell_state_dynamics/) under the [Allen Insitute for Cell Science Terms of Use](https://www.allencell.org/terms-of-use.html).
The data are also available via the AWS S3 API directly at `s3://allencell/aics/endo_cell_state_dynamics`.

Users can view image data without downloading it using [Vol-E](https://volumeviewer.allencell.org/).
Right click on any image data in BFF and select _Open with_ > _Vol-E_.
Users can interactively explore the data without downloading it using [Timelapse Feature Explorer](https://timelapse.allencell.org).

#### Access the [Endothelial cell state dynamics dataset]() on BFF

#### Explore [VE-cadherin segmentations for endothelial cell state dynamics]() on TFE

#### Explore [Grid-based patches for endothelial cell state dynamics]() on TFE

## Installation

This project requires Python 3.11.
We recommend using the most recent version of Python 3.11 (Python 3.11.12).
Package dependencies can be found in the `pyproject.toml` file.

### Installation using UV

We recommend using the Python package manager [uv](https://docs.astral.sh/uv) to manage dependencies and virtual environments.
Install uv following their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

**1. Navigate to where you want to clone this repository**

```bash
cd /path/to/directory/
```

**2. Clone the repo from GitHub**

```bash
git clone git@github.com:AllenCell/endothelial-pipeline.git
cd endothelial-pipeline
```

**3. Install the dependencies using uv**

For basic installation with just the core dependencies:

```bash
uv sync --no-dev
```

If you plan to develop code, you should also install the development dependencies:

```bash
uv sync
```

If you are on the Allen Institute for Cell Science local network, you can load on-prem data by installing `aicsfiles` (which is included in the optional `internal` dependency group):

```bash
uv sync --extra internal
```

**4. Activate the virtual environment**

Activate the virtual environment in the terminal:

For Windows:

```powershell
\path\to\venv\Scripts\activate
```

For Linux/Mac:

```bash
source /path/to/venv/bin/activate
```

You can deactivate the virtual environment using:

```
deactivate
```

### Alternative installation using `pip`

This project also includes a `requirements.txt` generated from the `uv.lock` file, which can be used to install requirements using `pip`.

> [!NOTE]
> This installation method will only install core dependencies.
> We recommend using uv to handle more complex installations of development and optional dependencies.

**1. Navigate to where you want to clone this repository**

```bash
cd /path/to/directory/
```

**2. Clone the repo from GitHub**

```bash
git clone git@github.com:AllenCell/endothelial-pipeline.git
cd endothelial-pipeline
```

**3. Create and activate a new virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**4. Install the dependencies using pip**

```bash
pip install -r requirements.txt
```

**5. Install the package**

```bash
pip install -e .
```

## Usage

All workflows in this repository can be run using the `endopipe` command.
Running the command without any arguments will provide a list of all available workflows.

```bash
uv run endopipe
```

You may want to use tags to filter these workflows to specific areas.

```bash
# Show all available tags
uv run endopipe -t

# Show only workflows with given tag
uv run endopipe -f TAG
```

Run workflows by passing the name of the workflow to the `endopipe` command.
All workflows have additional details on usage and arguments, which you can access by appending the `-h` or `--help` flag.

> [!IMPORTANT]
> Most workflows provided are designed to run in a high-performance computing setting or need the use of GPUs to run in a reasonable amount of time.
> To support review of the code, all production workflows have a "demo mode" (which you can access by appending the `-d` or `--demo-mode` flag) that modifies (e.g. by reducing the number of iterations) the workflow to run in a shorter amount of time.

```bash
# Run the workflow
uv run endopipe NAME-OF-WORKFLOW

# Show the workflow help message
uv run endopipe NAME-OF-WORKFLOW -h

# Run the workflow in demo mode
uv run endopipe NAME-OF-WORKFLOW -d
```

### Reproduce figures, tables, and movies

To reproduce figures, tables, and movies from the manuscript, run the corresponding workflow.
Some workflows should be run with an NVIDIA GPU, as indicated by the workflow help message.
Run the workflow with the GPU flag (`-g` or `--num-gpus`) to make sure GPUs are visible to the workflow.
All workflows will run without a GPU, but will be noticeably slower.

```bash
# Reproduce main figure N
uv run endopipe figure-N

# Reproduce supplemental figure N
uv run endopipe supp-figure-N

# Reproduce all supplemental movies
uv run endopipe supp-movies

# Reproduce all supplemental tables
uv run endopipe supp-tables
```
